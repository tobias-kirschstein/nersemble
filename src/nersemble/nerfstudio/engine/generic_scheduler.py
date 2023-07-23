import torch


class GenericScheduler(torch.nn.Module):
    """A generic scheduler"""

    def __init__(self, init_value, final_value, begin_step, end_step) -> None:
        super().__init__()
        self.init_value = init_value
        self.final_value = final_value
        self.begin_step = begin_step
        self.end_step = end_step

        self.value = final_value

    def update(self, step):
        if step > self.end_step:
            self.value = self.final_value
        elif step < self.begin_step:
            self.value = self.init_value
        else:
            delta = min(max((step - self.begin_step) / (self.end_step - self.begin_step), 0), 1) * (
                self.final_value - self.init_value
            )
            self.value = self.init_value + delta

    def get_value(self):
        if self.training:  # inherit from torch.nn.Module
            return self.value
        else:
            return self.final_value