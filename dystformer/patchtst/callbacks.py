import logging

from transformers import TrainerCallback, TrainerControl, TrainerState


class AdaptiveNumBinsCallback(TrainerCallback):
    def __init__(
        self,
        initial_bins: int,
        max_bins: int,
        step_interval: int,
        bin_delta: int,
        logger=None,
    ):
        """
        Args:
            initial_bins: initial number of bins for quantization
            max_bins: maximum number of bins for quantization
            step_interval: number of steps between each bin adjustment
            bin_delta: number of bins to adjust (add) by each step
        """
        self.num_bins = initial_bins
        self.max_bins = max_bins
        self.step_interval = step_interval
        self.bin_delta = bin_delta
        self.logger = logger or logging.getLogger(__name__)

    def on_step_begin(
        self, args, state: TrainerState, control: TrainerControl, **kwargs
    ):
        if self.num_bins is None:
            return

        if (state.global_step + 1) % self.step_interval == 0:
            if self.num_bins < self.max_bins:
                self.num_bins += self.bin_delta
                self.logger.info(
                    f"Adjusted num_bins to {self.num_bins} at step {state.global_step}"
                )
            else:
                self.logger.info(
                    f"Max num_bins reached at step {state.global_step}. Setting num_bins to None"
                )
                self.num_bins = None

        # Store num_bins in a shared state
        if not hasattr(state, "custom_state"):
            state.custom_state = {}  # type: ignore
        state.custom_state["num_bins"] = self.num_bins  # type: ignore

    def on_log(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # Log num_bins to the logger
        self.logger.info(
            f"Logging num_bins: {self.num_bins} at step {state.global_step}"
        )
        # If using wandb or tensorboard, log it there as well
        if args.report_to:
            for report in args.report_to:
                if report == "wandb":
                    import wandb

                    wandb.log({"num_bins": self.num_bins}, step=state.global_step)
                elif report == "tensorboard":
                    if control.should_log:
                        # control.log_history.append({"num_bins": self.num_bins})
                        if not hasattr(self, "log_history"):
                            self.log_history = []
                        self.log_history.append({"num_bins": self.num_bins})

    def get_num_bins(self):
        return self.num_bins
