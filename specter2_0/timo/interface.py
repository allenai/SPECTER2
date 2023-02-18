"""
This file contains the classes required by Semantic Scholar's
TIMO tooling.

You must provide a wrapper around your model, as well
as a definition of the objects it expects, and those it returns.
"""

from typing import List

from pydantic import BaseModel, BaseSettings, Field
import numpy as np
from enum import Enum
from transformers import AutoAdapterModel, AutoTokenizer
import os
import torch


class TaskType(Enum):
    DEFAULT = 1
    CLASSIFICATION = 2
    REGRESSION = 3
    PROXIMITY = 4
    ADHOC_QUERY = 5


class Instance(BaseModel):
    """
    Describes one Instance over which the model performs inference.

    The fields below are examples only; please replace them with
    appropriate fields for your model.

    To learn more about declaring pydantic model fields, please see:
    https://pydantic-docs.helpmanual.io/
    """

    title: str = Field(..., description="Title of the paper to be embedded/raw text query")
    abstract: str = Field(default=None, description="Abstract of the paper to be embedded", )
    task_type: TaskType = Field(..., description="Task format for which embedding is required. For no specific format, "
                                                 "pass DEFAULT. For adhoc search tasks, provide ADHOC_QUERY for "
                                                 "query and PROXIMITY for candidates.")


class Prediction(BaseModel):
    """
    Describes the outcome of inference for one Instance

    The fields below are examples only; please replace them with
    appropriate fields for your model.

    To learn more about declaring pydantic model fields, please see:
    https://pydantic-docs.helpmanual.io/
    """

    embedding: np.ndarray = Field(default_factory=lambda: np.zeros(768),
                                  description="Embedding for the paper(title, abstract) with dim. 768")

    class Config:
        arbitrary_types_allowed = True


class PredictorConfig(BaseSettings):
    """
    Configuration required by the model to do its work.
    Uninitialized fields will be set via Environment variables.

    The fields below are examples only; please replace them with ones
    appropriate for your model. These serve as a record of the ENV
    vars the consuming application needs to set.
    """

    # example_field: str = Field(default="asdf", description="Used to [...]")
    use_fp16: bool = Field(default=False, env="use_fp16", description="fp16 inference for embeddings")
    max_len: int = Field(default=512, description="max input length to be processed by the model")


class Predictor:
    """
    Interface on to your underlying model.

    This class is instantiated at application startup as a singleton.
    You should initialize your model inside of it, and implement
    prediction methods.

    If you specified an artifacts.tar.gz for your model, it will
    have been extracted to `artifacts_dir`, provided as a constructor
    arg below.
    """

    _config: PredictorConfig
    _artifacts_dir: str

    def __init__(self, config: PredictorConfig, artifacts_dir: str):
        self._config = config
        self._artifacts_dir = artifacts_dir
        self.base_encoder = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self) -> None:
        """
        Perform whatever start-up operations are required to get your
        model ready for inference. This operation is performed only once
        during the application life-cycle.
        """
        self.base_encoder = AutoAdapterModel.from_pretrained(f"{self._artifacts_dir}/base")
        self.tokenizer = AutoTokenizer.from_pretrained(f"{self._artifacts_dir}/base")
        adapter_dir = f"{self._artifacts_dir}/adapters"
        adapter_names = os.listdir(adapter_dir)
        for aname in adapter_names:
            self.base_encoder.load_adapter(f"{adapter_dir}/{aname}/", load_as=aname)

        if torch.cuda.is_available():
            self.base_encoder.to('cuda')
        self.base_encoder.eval()

    def encode_batch(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, task_type: TaskType) -> np.ndarray:
        """
                Perform whatever start-up operations are required to get your
                model ready for inference. This operation is performed only once
                during the application life-cycle.
        """
        if TaskType.DEFAULT != task_type:
            self.base_encoder.base_model.set_active_adapters(task_type.name.lower())
        else:
            self.base_encoder.base_model.set_active_adapters(None)
        output = self.base_encoder(input_ids, attention_mask=attention_mask)
        emb_tensor = output.last_hidden_state[:, 0, :]
        return emb_tensor.detach().cpu().numpy()

    def predict_one(self, instance: Instance) -> Prediction:
        """
        Produces embeddings in a Prediction object for a single input paper Instance.
        The title and abstract are first concatenated and tokenized.
        The tokenized input is then provided to the model to generate the final embedding.
        """
        title = instance.title
        abstract = instance.abstract
        text = f"{title} {self.tokenizer.sep_token} {abstract}" if abstract else title
        print(text)
        input_ids = self.tokenizer([text], padding=True, truncation=True,
                                   return_tensors="pt", return_token_type_ids=False, max_length=self._config.max_len)
        embedding = self.encode_batch(task_type=instance.task_type, **input_ids)
        embedding = embedding.astype(np.float16) if self._config.use_fp16 else embedding
        return Prediction(embedding=embedding)

    def predict_batch(self, instances: List[Instance]) -> List[Prediction]:
        """
        Method called by the client application. One or more Instances will
        be provided, and the caller expects a corresponding Prediction for
        each one.

        If your model gets performance benefits from batching during inference,
        implement that here, explicitly.

        Otherwise, you can leave this method as-is and just implement
        `predict_one()` above. The default implementation here passes
        each Instance into `predict_one()`, one at a time.

        The size of the batches passed into this method is configurable
        via environment variable by the calling application.
        """
        task_types = np.array([ins.task_type.value for ins in instances])
        task_idx_map = {ttype: np.where(task_types == ttype) for ttype in np.unique(task_types)}
        text_batch = [f"{ins.title} {self.tokenizer.sep_token} {ins.abstract}" if ins.abstract else ins.title for ins in
                      instances]
        input_ids = self.tokenizer(text_batch, padding=True, truncation=True,
                                   return_tensors="pt", return_token_type_ids=False, max_length=self._config.max_len)
        batch_embeddings = np.zeros((len(instances), self.base_encoder.config.hidden_size))
        input_ids, attention_mask = input_ids["input_ids"], input_ids["attention_mask"]
        for ttype in np.unique(task_types):
            sub_input_ids = {"input_ids": input_ids[task_idx_map[ttype]],
                             "attention_mask": attention_mask[task_idx_map[ttype]]}
            sub_embedding = self.encode_batch(task_type=TaskType(ttype), **sub_input_ids, )
            batch_embeddings[task_idx_map[ttype]] = sub_embedding
        batch_embeddings = batch_embeddings.astype(np.float16) if self._config.use_fp16 else batch_embeddings
        return [Prediction(embedding=emb) for emb in batch_embeddings]
