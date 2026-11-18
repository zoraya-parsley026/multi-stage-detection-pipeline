from pathlib import Path
from typing import Any

import yaml


class BaseModelConfig:

    def __init__(
        self,
        *,
        input: dict[str, Any] | None = None,
        output: dict[str, Any] | None = None,
       
        preprocess: dict[str, Any] | None = None,
        postprocess: dict[str, Any] | None = None,
       
        meta: dict[str, Any] | None = None,
       
        **extensions: Any,
    ) -> None:
        sections = {
            "input": input, "output": output,
            "preprocess": preprocess, "postprocess": postprocess, 
            "meta": meta,
        }

        for name, section in sections.items():
            if section is not None and not isinstance(section, dict):
                raise ValueError(f"Configuration section '{name}' must be a mapping.")
            
        self.input = dict(input or {})
        self.output = dict(output or {})
        
        self.preprocess = dict(preprocess or {})
        self.postprocess = dict(postprocess or {})
        
        self.metadata = dict(meta or {})

        self.extensions = extensions

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BaseModelConfig":
        with Path(path).open(encoding="utf-8") as file:
            data = yaml.safe_load(file)
        if data is None:
            data = {}
        if not isinstance(data, dict) or not all(isinstance(key, str) for key in data):
            raise ValueError("Model configuration YAML must contain a mapping with string keys.")
        return cls(**data)
