import sys, time, torch
from utils.training_ops import Average, accuracy, loader_accuracy

# Make the training / testing loaders
from utils.datasets import get_cifar100
train_loader, test_loader = get_cifar100(batch_size=128)
mu, std = [n/255. for n in [129.3, 124.1, 112.4]], [n/255. for n in [68.2,  65.4,  70.4]]

# Define the model
from solvers.fixed_step import FixedStepSolver, FixedStepRK4RegSolver, FixedStepNumericalLyapunovSolver, FixedStepNumericalLyapunovSolverV2
from integrators.simple import Euler, ModifiedEuler, RungeKutta4, K2K3L2DistRungeKutta4
from networks.odenet_2 import OdeNet

# Load Adversarial attacks
from utils.fgsm import fgsm_attack
from utils.deepfool import deepfool_attack
from utils.pgd import pgd_attack
from utils.attacks import gaussian_noise_attack, salt_and_pepper_attack, simba_attack

# Define experiments
experiments = [
	# Unregularized baseline
	(Euler, FixedStepSolver, 2, None, None),
	(Euler, FixedStepSolver, 2, None, None),
	(Euler, FixedStepSolver, 2, None, None),
	# Lyapunov exponents models
	(Euler, FixedStepNumericalLyapunovSolverV2, 2, 1e-2, 1),
	(Euler, FixedStepNumericalLyapunovSolverV2, 2, 1e-2, 1),
	(Euler, FixedStepNumericalLyapunovSolverV2, 2, 1e-2, 1),
	(Euler, FixedStepNumericalLyapunovSolverV2, 2, 1e-2, 2),
	(Euler, FixedStepNumericalLyapunovSolverV2, 2, 1e-2, 2),
	(Euler, FixedStepNumericalLyapunovSolverV2, 2, 1e-2, 2),
	(Euler, FixedStepNumericalLyapunovSolverV2, 2, 1e-2, 3),
	(Euler, FixedStepNumericalLyapunovSolverV2, 2, 1e-2, 3),
	(Euler, FixedStepNumericalLyapunovSolverV2, 2, 1e-2, 3),
	# Runge Kutta divergence models
	(K2K3L2DistRungeKutta4, FixedStepRK4RegSolver, 1, 1e-2, None),
	(K2K3L2DistRungeKutta4, FixedStepRK4RegSolver, 1, 1e-2, None),
	(K2K3L2DistRungeKutta4, FixedStepRK4RegSolver, 1, 1e-2, None),
]

for intr, sol, steps, le_reg, vectors in experiments:
	for i in range(3):
		# Define the model
		model = OdeNet(solver=sol, integrator=intr, t0=0.0, t1=1.0, steps=steps, in_channels=3, channels=[64, 128, 256, 512], classes=100,  mu=mu, std=std)
		model.solver1.n, model.solver2.n, model.solver3.n, model.solver4.n = vectors, vectors, vectors, vectors

		# Naming for saving
		post_fix = "%sStep-leReg%s-Vecs%s-Run%s" % (str(steps), str(le_reg), str(vectors), str(i))
		print("\n" + model.solver1.name + "-" + model.solver1.method.name + "-" + post_fix)

		# Define the convenient average calculators
		time_keeper, train_acc, train_exps, test_acc, test_exps = Average(), Average(), Average(), Average(), Average()

		epochs = 20
		best_acc = 0
		for e in range(epochs):
			# Train loop
			model.train()
			for i, (data, target) in enumerate(train_loader):
				s = time.time()
				# Convert data proper device, forward pass and calculate loss
				data, target = data.to(model.device), target.to(model.device)
				pred = model(data)
				ce_loss = model.loss_module(pred, target)
				
				#Take optimizer step
				model.optimizer.zero_grad()
				if le_reg and model.exps:
					(ce_loss + le_reg * model.exps).backward()
				else:
					ce_loss.backward()
				model.optimizer.step()

				# Update metrics
				time_keeper.update(time.time() - s)
				train_acc.update(accuracy(pred, target))
				if le_reg and model.exps:
					train_exps.update(model.exps.item())
			
			# Evaluation loop
			with torch.no_grad():
				for i, (data, target) in enumerate(test_loader):
					data, target = data.to(model.device), target.to(model.device)
					pred = model(data)
					ce_loss = model.loss_module(pred, target)
					test_acc.update(accuracy(pred, target))
					if le_reg and model.exps:
						test_exps.update(model.exps.item())
			print('Epoch: %d / %d | Time per iter %.3f | Train acc: %.3f | Train exps: %.3f | Test acc: %.3f | Test exps: %.3f' % (e + 1, epochs, time_keeper.eval(), 100 * train_acc.eval(), train_exps.eval(), 100 * test_acc.eval(), test_exps.eval()))

			# Save the model
			if test_acc.eval() >= best_acc:
				torch.save({'state_dict': model.state_dict()}, 'checkpoints/cifar100/' + model.solver1.name + "-" + model.solver1.method.name + "-" + post_fix + '.pth')
				best_acc = test_acc.eval()

			# Reset statistics each epoch:
			time_keeper.reset(), train_acc.reset(), train_exps.reset(), test_acc.reset(), test_exps.reset()

			# Decay Learning Rate
			model.scheduler.step()

		# Perform FGSM tests
		# Check baseline accuracy
		gaussian_noise_attack(model, test_loader, std=0.0)

		# Run the random gaussian attack
		gaussian_noise_attack(model, test_loader, std=35/255.)

		# Run salt and pepper attack
		salt_and_pepper_attack(model, test_loader, 0.1)

		# Run test for each epsilon
		fgsm_attack(model, test_loader, 0.2/255.)

		# Run the PGD attack
		pgd_attack(model, test_loader, epsilon=4/255., alpha=1/255., iters=20)

		# Run DeepFool tests
		deepfool_attack(model, test_loader, 100)

		# Run black box attack
		simba_attack(model, test_loader)