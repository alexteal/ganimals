# put this file in the same directory as main.py and add the following line to main.py:
# from SlantedTriangularLR import SlantedTriangularLR

# data_loader is passed to estimate total number of steps
# INSTEAD OF MOST SCHEDULERS, YOU MUST CALL THIS EVERY STEP!, i.e. move scheduler.step() from epoch loop to step loop
# cut_frac the fraction of steps to ramp up learning rate. (e.g. 0.1 = first 10% of steps has positive slope of lr)
# ratio is the ratio of the minimum learning rate to the maximum learning rate
# if your highest learning rate (lr_max) is 0.001 and your ratio is 32, your lowest learning rate will be 0.001/32
# hi Jack here's what i recommend for the initialization:
# scheduler = SlantedTriangularLR(optimizer, 0.1, 32,
#          2 * 0.0005 * (batch_size / 512), len(train_loader) * num_epochs)
#          [or however else you can calculate total number of steps]
class SlantedTriangularLR:
    def __init__(self, optimizer, cut_frac, ratio, lr_max, total_num_steps):
        self.optimizer = optimizer
        self.cut_frac = cut_frac
        self.ratio = ratio
        self.lr_max = lr_max
        self.total_iterations = total_num_steps
        self.cut = int(self.total_iterations * cut_frac)
        self.current_iteration = 0

    def step(self):
        self.current_iteration += 1

        p = self.current_iteration / self.total_iterations
        if self.current_iteration < self.cut:
            lr = self.lr_max * p
        else:
            lr = self.lr_max * (1 + (self.cut - self.current_iteration) / (self.total_iterations - self.cut) * (
                    self.ratio - 1)) / self.ratio

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        if self.current_iteration % 10 == 0:
            print(f"Learning rate at step {self.current_iteration}: {lr}")
