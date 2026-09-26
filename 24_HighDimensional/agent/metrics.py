import torch


class StreamingMetrics:
    def __init__(self):
        self.n = 0 
        self.count = 0
        self.ss_res = 0.0
        self.sum_y = None 
        self.sum_y2 = 0.0
        self.mag_sq_err = 0.0
        self.mag_count = 0

    @torch.no_grad()
    def update(self, pred, target):
        pred = pred.double()
        target = target.double()

        b = target.shape[0]
        self.n += b
        self.count += target.numel()

        self.ss_res += torch.sum((pred - target) ** 2).item()
        self.sum_y2 += torch.sum(target ** 2).item()

        batch_sum = target.sum(dim=0)
        if self.sum_y is None:
            self.sum_y = batch_sum
        else:
            self.sum_y += batch_sum

        pred_mag_sq = pred[:, 0] ** 2 + pred[:, 1] ** 2
        target_mag_sq = target[:, 0] ** 2 + target[:, 1] ** 2
        self.mag_sq_err += torch.sum(
            (torch.sqrt(pred_mag_sq) - torch.sqrt(target_mag_sq)) ** 2
        ).item()
        self.mag_count += pred_mag_sq.numel()

    def compute(self, eps=1e-12):
        if self.count == 0:
            raise RuntimeError("StreamingMetrics.compute() called before any update()")

        rmse = (self.ss_res / self.count) ** 0.5

 

        return {"rmse": rmse}


def eval_metrics(pred, target):
    acc = StreamingMetrics()
    acc.update(pred, target)
    return acc.compute()