import os
import subprocess
import tarfile
from collections import defaultdict
from glob import glob

import faiss
import numpy as np
import pyarrow as pa
import requests
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src.models._api import register_model

__all__ = ["CLIP"]

_DATABASES = {
    "cc12m": "1HyM4mnKSxF0sqzAe-KZL8y-cQWRPiuXn",
}


class MetadataProvider:
    """Metadata provider for the retrieval database.

    Args:
    ----
        metadata_dir (str): Path to the metadata directory.

    """

    def __init__(self, metadata_dir: str) -> None:
        metadatas = [
            str(a) for a in sorted(glob(os.path.join(metadata_dir, "*"))) if os.path.isfile(a)
        ]
        self.table = pa.concat_tables(
            [
                pa.ipc.RecordBatchFileReader(pa.memory_map(metadata, "r")).read_all()
                for metadata in metadatas
            ]
        )

    def get(self, ids: list[int]) -> dict:
        """Get the metadata for the given ids.

        Args:
        ----
            ids (list): List of ids.

        """
        columns = self.table.schema.names
        end_ids = [i + 1 for i in ids]
        t = pa.concat_tables(
            [self.table[start:end] for start, end in zip(ids, end_ids, strict=False)]
        )
        return t.select(columns).to_pandas().to_dict("records")


@register_model(name="clip")
class CLIP(torch.nn.Module):
    """CLIP retrieval model.

    Args:
    ----
        model_name_or_path (str): Path to pretrained model or model identifier from
            huggingface.co/models. Defaults to "openai/clip-vit-large-patch14".
        device (str): Device type for model weights. Defaults to "cpu".
        device_map (str): Device map for model parallel loading. Defaults to "auto".
        dtype (str | torch.dtype): Data type for model weights. Defaults to "torch.float16".
        **kwargs (dict): Extra model arguments.

    """

    def __init__(
        self,
        model_name_or_path: str = "openai/clip-vit-large-patch14",
        device: str = "cpu",
        device_map: str = "auto",
        dtype: str | torch.dtype = "float16",
        **kwargs,
    ) -> None:
        super().__init__()

        if isinstance(dtype, str) and dtype != "auto":
            dtype = getattr(torch, dtype)

        database_name = kwargs.get("database_name", "cc12m")
        file_gdown_id = _DATABASES[database_name]
        target_dir = ".cache/data"
        os.makedirs(target_dir, exist_ok=True)

        file_path = os.path.join(target_dir, database_name + ".tar.gz")
        if not os.path.exists(os.path.join(target_dir, database_name)):
            subprocess.run(["uvx", "gdown", "--id", file_gdown_id, "-O", file_path])  # noqa: S603, S607  # nosec B607
            with tarfile.open(file_path, "r:gz") as f:
                f.extractall(target_dir)  # noqa: S202  # nosec B202
            os.remove(file_path)

        self._model_name_or_path = model_name_or_path
        self._num_samples = kwargs.get("num_samples", 10)
        self.device = torch.device(device)
        self.device_map = device_map
        self.dtype = dtype

        self.model = CLIPModel.from_pretrained(self._model_name_or_path)
        self.processor = CLIPProcessor.from_pretrained(self._model_name_or_path)
        self.model.to(device=self.device, dtype=self.dtype)

        if model_name_or_path == "openai/clip-vit-large-patch14":
            database_path = os.path.join(target_dir, database_name, "vit-l-14")
            image_index_path = os.path.join(database_path, "image.index")
            text_index_path = os.path.join(database_path, "text.index")

            self._image_index = (
                faiss.read_index(image_index_path, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
                if os.path.exists(image_index_path)
                else None
            )
            self._text_index = (
                faiss.read_index(text_index_path, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
                if os.path.exists(text_index_path)
                else None
            )

            metadata_dir = os.path.join(database_path, "metadata")
            self._metadata_provider = MetadataProvider(metadata_dir)
        else:
            raise ValueError(f"Unsupported model_name_or_path {model_name_or_path}")

    @torch.no_grad()
    def __call__(self, image_path: str) -> str:
        """Caption the image.

        Args:
        ----
            image_path (str): The path to the input image.

        """
        if image_path.startswith(("http://", "https://")):
            image = Image.open(requests.get(image_path, stream=True, timeout=3000).raw)
        else:
            image = Image.open(image_path)

        inputs = self.processor(
            images=image,
            text=["cat", "dog"],
            return_tensors="pt",
        ).to(self.device)
        outputs = self.model(**inputs)

        image_embeds = outputs.image_embeds
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        image_embeds = image_embeds.cpu().detach().numpy().tolist()

        if isinstance(image_embeds[0], float):
            image_embeds = [image_embeds]

        query = np.array(image_embeds).astype("float32")
        results = self._query(query, modality="text", num_samples=self._num_samples)

        captions = [[r["caption"] for r in result] for result in results]
        return captions

    async def ainvoke(self, user_input: dict) -> dict:
        """Run the retrieval model on the user input.

        Args:
        ----
            user_input (dict): The user input, containing the image for retrieval.

        """
        message = user_input["messages"][-1]

        conversation = None
        if isinstance(message[1], str) == 1:
            conversation = message[1]
        elif isinstance(message[1], tuple):
            content_type, content = message[1]

            if content_type != "image":
                raise ValueError(
                    f"Invalid type for model message. Expected 'image', got '{content_type}'."
                )

            conversation = content
        elif isinstance(message[1], list):
            for content_type, content in message[1]:
                if content_type != "image":
                    continue

                if conversation is not None:
                    raise ValueError("Multiple input image passed to the model. Expected one.")

                conversation = content
        else:
            raise ValueError(f"Too many values in the content tuple {message[1]}")

        captions = self.__call__(conversation)[0]
        text = "\n".join(captions)

        user_input["messages"].append(("clip", text))
        return user_input

    def to(self, *args, **kwargs) -> "CLIP":
        """Perform Tensor dtype and/or device conversion.

        Args:
        ----
            *args: Arguments.
            **kwargs: Keyword arguments.

        """
        device, dtype, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        self.device = device
        self.dtype = dtype
        super().to(*args, **kwargs)

    def _deduplicate_embeddings(self, embeddings: np.ndarray, threshold: float = 0.94) -> set:
        """Deduplicate embeddings.

        Args:
        ----
            embeddings (np.ndarray): Embeddings to deduplicate.
            threshold (float): Threshold to use for deduplication. Default is 0.94.

        """
        embeddings = np.asarray(embeddings).astype("float32")
        faiss.normalize_L2(embeddings)  # For cosine similarity
        dim = embeddings.shape[1]

        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        k = 10  # Top-k neighbors
        D, I = index.search(embeddings, k)  # noqa: E741

        parent = list(range(len(embeddings)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i in range(len(embeddings)):
            for j, sim in zip(I[i], D[i], strict=False):
                if i == j:
                    continue
                if sim >= threshold:
                    union(i, j)

        root_map = defaultdict(list)
        for i in range(len(embeddings)):
            root_map[find(i)].append(i)

        non_uniques = set()
        for group in root_map.values():
            for dup in group[1:]:
                non_uniques.add(dup)

        return non_uniques

    def _get_connected_components(self, neighbors: dict) -> list:
        """Find connected components in a graph.

        Args:
        ----
            neighbors (dict): Dictionary of neighbors.

        """
        seen = set()

        def component(node: list) -> list:
            r = []
            nodes = {node}
            while nodes:
                node = nodes.pop()
                seen.add(node)
                nodes |= set(neighbors[node]) - seen
                r.append(node)
            return r

        u = []
        for node in neighbors:
            if node not in seen:
                u.append(component(node))
        return u

    def _map_to_metadata(
        self, indices: list, distances: list, embs: list, num_images: int
    ) -> list:
        """Map the indices to metadata.

        Args:
        ----
            indices (list): List of indices.
            distances (list): List of distances.
            embs (list): List of results embeddings.
            num_images (int): Number of images.

        """
        results = []
        metas = self._metadata_provider.get(indices[:num_images])
        for key, (d, i, emb) in enumerate(zip(distances, indices, embs, strict=False)):
            output = {}
            meta = None if key + 1 > len(metas) else metas[key]
            if meta is not None:
                output.update(self._meta_to_dict(meta))
            output["id"] = i.item()
            output["similarity"] = d.item()
            output["sample_z"] = emb.tolist()
            results.append(output)

        return results

    def _meta_to_dict(self, metadata: dict) -> dict:
        """Convert metadata to dict.

        Args:
        ----
            metadata (dict): Metadata.

        """
        output = {}
        for k, v in metadata.items():
            if isinstance(v, bytes):
                v = v.decode()
            elif type(v).__module__ == np.__name__:
                v = v.item()
            output[k] = v
        return output

    def _query(
        self, query: np.ndarray, modality: str = "text", num_samples: int = 10
    ) -> list[list[dict]]:
        """Query the database.

        Args:
        ----
            query (np.ndarray): Query to search.
            modality (str): Modality to search. One of `image` or `text`. Default to `text`.
            num_samples (int): Number of samples to return. Default is 40.

        """
        index = self._image_index if modality == "image" else self._text_index

        distances, indices, embeddings = index.search_and_reconstruct(query, num_samples)
        results = [indices[i] for i in range(len(indices))]

        nb_results = [np.where(r == -1)[0] for r in results]
        total_distances = []
        total_indices = []
        total_embeddings = []
        for i in range(len(results)):
            num_res = nb_results[i][0] if len(nb_results[i]) > 0 else len(results[i])

            result_indices = results[i][:num_res]
            result_distances = distances[i][:num_res]
            result_embeddings = embeddings[i][:num_res]

            # Normalize embeddings
            l2 = np.atleast_1d(np.linalg.norm(result_embeddings, 2, -1))
            l2[l2 == 0] = 1
            result_embeddings = result_embeddings / np.expand_dims(l2, -1)

            # Deduplicate embeddings
            local_indices_to_remove = self._deduplicate_embeddings(result_embeddings)
            indices_to_remove = set()
            for local_index in local_indices_to_remove:
                indices_to_remove.add(result_indices[local_index])

            curr_indices = []
            curr_distances = []
            curr_embeddings = []
            for ind, dis, emb in zip(
                result_indices, result_distances, result_embeddings, strict=False
            ):
                if ind not in indices_to_remove:
                    indices_to_remove.add(ind)
                    curr_indices.append(ind)
                    curr_distances.append(dis)
                    curr_embeddings.append(emb)

            total_indices.append(curr_indices)
            total_distances.append(curr_distances)
            total_embeddings.append(curr_embeddings)

        if len(total_distances) == 0:
            return []

        total_results = []
        for i in range(len(total_distances)):
            results = self._map_to_metadata(
                total_indices[i], total_distances[i], total_embeddings[i], num_samples
            )
            total_results.append(results)

        return total_results
