import torch

class FixedStepSolver():
    def __init__(self, f, t0, t1, method, steps=None, step_size=None, verbose=False):
        self.name = "FixedStepSolver"
        self.f = f
        self.t0 = t0
        self.t1 = t1
        self.method = method
        self.exp = None

        # Define the step sizes h to go from t0 to t1
        assert steps or step_size, "Either steps or step size should be defined!"
        if steps:
            self.hs = [(t1 - t0) / steps for s in range(steps)]
        else:
            assert step_size <= (t1 - t0), "Step size should be smaller than integration time!"
            self.hs = [step_size for _ in range(int((t1 - t0) / step_size))]
            # Add the residual in the last step, if required
            if (t1 - t0) % step_size != 0:
                self.hs.append((t1 - t0) - sum(self.hs))
        if verbose:
            print("This solver will be using the following time deltas:", self.hs)
            print("This solver will require", self.method.order * len(self.hs), "gradient evaluations")

    def integrate(self, y, reset=False):
        # For every step h, we integrate using the given method, starting from t0, y
        t = self.t0
        for h in self.hs:
            y = self.method.step(self.f, t, h, y)
            t += h
        return y

class FixedStepRK4RegSolver():
    def __init__(self, f, t0, t1, method, steps=None, step_size=None, verbose=False):
        self.name = "FixedStepRK4RegSolver"
        self.f = f
        self.t0 = t0
        self.t1 = t1
        self.method = method

        # Define the step sizes h to go from t0 to t1
        assert steps or step_size, "Either steps or step size should be defined!"
        if steps:
            self.hs = [(t1 - t0) / steps for s in range(steps)]
        else:
            assert step_size <= (t1 - t0), "Step size should be smaller than integration time!"
            self.hs = [step_size for _ in range(int((t1 - t0) / step_size))]
            # Add the residual in the last step, if required
            if (t1 - t0) % step_size != 0:
                self.hs.append((t1 - t0) - sum(self.hs))
        if verbose:
            print("This solver will be using the following time deltas:", self.hs)
            print("This solver will require", self.method.order * len(self.hs), "gradient evaluations")

    def integrate(self, y, reset=False):
        # For every step h, we integrate using the given method, starting from t0, y
        t = self.t0
        self.exp = 0
        for h in self.hs:
            y, k2k3dist = self.method.step(self.f, t, h, y)
            self.exp += k2k3dist
            t += h
        return y

class FixedStepNumericalLyapunovSolver():
    def __init__(self, f, t0, t1, method, steps=None, step_size=None, verbose=False, eps=1e-6, n=2, upsample=None):
        self.name = "FixedStepNumericalLyapunovSolver"
        self.f = f
        self.t0 = t0
        self.t1 = t1
        self.method = method
        self.eps, self.n = eps, n
        self.vecs = [None]
        self.upsample = upsample

        # Define the step sizes h to go from t0 to t1
        assert steps or step_size, "Either steps or step size should be defined!"
        if steps:
            self.hs = [(t1 - t0) / steps for s in range(steps)]
        else:
            assert step_size <= (t1 - t0), "Step size should be smaller than integration time!"
            self.hs = [step_size for _ in range(int((t1 - t0) / step_size))]
            # Add the residual in the last step, if required
            if (t1 - t0) % step_size != 0:
                self.hs.append((t1 - t0) - sum(self.hs))
        if verbose:
            print("This solver will be using the following time deltas:", self.hs)
            print("This solver will require", self.method.order * len(self.hs), "gradient evaluations")

    def reset_vec(self, y):
        # Random vector and its norm along the batch size
        ps = [torch.randn_like(y[0,...]) for _ in range(self.n)]
        ps_norm = [torch.norm(p, p=2) for p in ps]
        self.vecs = [p / pn for p, pn in zip(ps, ps_norm)]

    def integrate(self, y, reset=False):
        t = self.t0
        # First we make N versions of y. Either by taking a step in the direction of the Lyapunov vector, else randomly
        if reset or not isinstance(self.vecs[0], torch.Tensor):
            self.reset_vec(y)
        ys = [y + self.eps * self.vecs[i] for i in range(self.n)]

        # Concatenate the y to batch the vectors
        yb = torch.cat([y] + ys, dim=0)

        # Upsample the batch if necessary
        if isinstance(self.upsample, torch.nn.Sequential):
            yb = self.upsample(yb)

        # Integrate the system for both y and ys
        for h in self.hs:
            yb = self.method.step(self.f, t, h, yb)
            t += h
        yl = torch.chunk(yb, self.n + 1, dim=0)
        
        # Calculate the seperation
        self.exp = 0
        for i in range(self.n):
            # Calculate the difference minus the projection of earlier vectors (ealier vector, scaled by the dot product of yl and earlier vectors)
            #print(i, "Before: ", (yl[i + 1] - yl[0]).mean(0).norm())
            diff = yl[i + 1] - yl[0]
            proj = sum([self.vecs[j][None, ...].repeat(y.shape[0],1,1,1) * (diff * self.vecs[j]).sum([1,2,3])[:, None, None, None] for j in range(0, i)])
            diff = (diff - proj).mean(0)
            #print(i, "After: ", diff.norm())
            
            # Normalize the difference
            norm_diff = torch.norm(diff, p=2)

            # The Lyapunov or Bred vector is the normalized vector in the difference direction
            self.vecs[i] = (diff / norm_diff).detach()                     #torch.where(diff != 0, diff / norm_diff, torch.zeros_like(diff)).detach()
            self.exp += 1 / t * torch.log(norm_diff / self.eps + 1e-12)    #to avoid log(0 / eps)
        return yl[0]

class FixedStepNumericalLyapunovSolverV2():
    def __init__(self, f, t0, t1, method, steps=None, step_size=None, verbose=False, eps=1, n=2):
        self.name = "FixedStepNumericalLyapunovSolverV2"
        self.f = f
        self.t0 = t0
        self.t1 = t1
        self.method = method
        self.eps, self.n = eps, n
        self.vecs = [None]
        self.loops = 3
        self.diff = [None for _ in range(n)]
        self.lyapunov = True

        # Define the step sizes h to go from t0 to t1
        assert steps or step_size, "Either steps or step size should be defined!"
        if steps:
            self.hs = [(t1 - t0) / steps for s in range(steps)]
        else:
            assert step_size <= (t1 - t0), "Step size should be smaller than integration time!"
            self.hs = [step_size for _ in range(int((t1 - t0) / step_size))]
            # Add the residual in the last step, if required
            if (t1 - t0) % step_size != 0:
                self.hs.append((t1 - t0) - sum(self.hs))
        if verbose:
            print("This solver will be using the following time deltas:", self.hs)
            print("This solver will require", self.method.order * len(self.hs), "gradient evaluations")

    def batch_normalize(self, p):
        return p / torch.norm(p.view(p.size(0), -1), p=2, dim=1)[:, None, None, None]

    def reset_vec(self, y):
        # Random vector and its norm along the batch size
        ps = [torch.randn_like(y) for _ in range(self.n)]
        self.vecs = [self.batch_normalize(p) for p in ps]

    def integrate(self, y, reset=False):
        # First we integrate the original system y
        t = self.t0
        for h in self.hs:
            y = self.method.step(self.f, t, h, y)
            t += h

        if self.lyapunov:
            # Then we iteratively determine the Lyapunov vector for the batch
            self.exp = 0
            
            # First we make N versions of y by randomly mutating
            self.reset_vec(y)
            self.diff = []
            for l in range(self.loops):
                # Re-do the orbit seperation each loop
                ys = [y + self.eps * self.vecs[i] for i in range(self.n)]

                # Concatenate the y to batch the vectors
                yb = torch.cat(ys, dim=0)
                
                # Integrate the system for both y and ys
                t = self.t0
                for h in self.hs:
                    yb = self.method.step(self.f, t, h, yb)
                    t += h
                yl = torch.chunk(yb, self.n, dim=0)
                
                # Calculate the seperation
                
                for i in range(self.n):
                    # Calculate the difference minus the projection of earlier vectors (ealier vector, scaled by the dot product of yl and earlier vectors)
                    diff = yl[i] - y
                    proj = sum([self.vecs[j] * (diff * self.vecs[j]).sum([1,2,3])[:, None, None, None] for j in range(0, i)])
                    diff = (diff - proj)
                
                if l < self.loops - 1:
                    # The Lyapunov is the normalized vector in the difference direction
                    self.vecs[i] = self.batch_normalize(diff).detach()
                else:
                    self.diff.append(diff)
                    # Calculate exponent at last divergence
                    self.exp += 1 / t * torch.log(torch.norm(diff.view(diff.size(0), -1), p=2, dim=1) / self.eps + 1e-10).mean()
        else:
            self.exp = None
        return y
