example of using ConfigABC:
@dataclass
class OptimizerConfig(ConfigABC):
    optimizer_type: str = "Adam"
    optimizer_kwargs: dict = field(default_factory=dict)
    lr_scheduler_type: Optional[str] = None
    lr_scheduler_kwargs: dict = field(default_factory=dict)

    def create(self, params):
        optimizer = getattr(optim, self.optimizer_type)(params, **self.optimizer_kwargs)
        lr_scheduler = None
        if self.lr_scheduler_type is not None:
            lr_scheduler = getattr(optim.lr_scheduler, self.lr_scheduler_type)(optimizer, **self.lr_scheduler_kwargs)
        return optimizer, lr_scheduler

a command line is automatically generated with the following line:
config = OptimizerConfig.get_argparser().parse_args()