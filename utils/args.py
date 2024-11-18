from dataclasses import dataclass, field


@dataclass
class SFTDataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})