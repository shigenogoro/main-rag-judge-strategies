from __future__ import annotations
import re
import string
import random
import os, json, subprocess
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Sequence
from dataclasses import dataclass
import time

from tqdm import tqdm

from beir import util as beir_util
from beir.datasets.data_loader import GenericDataLoader
from pyserini.search.lucene import LuceneSearcher
from pyserini.index.lucene import LuceneIndexReader

try:
    import pytrec_eval
except Exception:
    pytrec_eval = None

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

import pandas as pd
from huggingface_hub import login

import matplotlib.pyplot as plt



@dataclass
class BeirBM25Config:
    dataset_name: str
    split: str = "test"
    work_root: Path = Path("work")
    index_threads: int = 16
    k: int = 100
    threads: int = 32
    k1: float = 0.9
    b: float = 0.4
    run_tag: str = "bm25"
    include_text_in_cache: bool = True


class BeirBM25Runner:
    """Download → index → retrieve → (eval) → write trecrun + rag cache."""
    def __init__(self, cfg: BeirBM25Config):
        self.cfg = cfg
        self.ds_root = Path(cfg.work_root) / cfg.dataset_name
        self.ds_root.mkdir(parents=True, exist_ok=True)
        self.data_dir  = self.ds_root / "dataset"
        self.jsonl_dir = self.ds_root / "json_for_pyserini"
        self.index_dir = self.ds_root / "index_bm25"
        self.out_dir = self.ds_root / "runs"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.corpus_file: Optional[Path] = None
        self.queries_file: Optional[Path] = None
        self.qrels_file: Optional[Path] = None

    def download_dataset(self) -> str:
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{self.cfg.dataset_name}.zip"
        data_path = beir_util.download_and_unzip(url, out_dir=str(self.data_dir))
        return data_path

    def _locate_files(self, data_path: str):
        base = Path(data_path)
        corpus = base / "corpus.jsonl"
        if not corpus.exists():
            raise FileNotFoundError(f"[{self.cfg.dataset_name}] corpus.jsonl missing in {base}")
        q_candidates = [base / f"queries.{self.cfg.split}.jsonl", base / "queries.jsonl"]
        queries = next((p for p in q_candidates if p.exists()), None)
        if queries is None:
            raise FileNotFoundError(f"[{self.cfg.dataset_name}] queries jsonl missing in {base}")
        qrels = None
        qrels_dir = base / "qrels"
        for ext in ("tsv", "txt"):
            p = qrels_dir / f"{self.cfg.split}.{ext}"
            if p.exists():
                qrels = p; break
        self.corpus_file, self.queries_file, self.qrels_file = corpus, queries, qrels

    def _write_json_for_pyserini(self, corpus: Dict[str, Dict]):
        if self.jsonl_dir.exists() and any(self.jsonl_dir.iterdir()):
            print(f"[convert] exists: {self.jsonl_dir}"); return
        self.jsonl_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.jsonl_dir / "docs_00.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for cid, rec in tqdm(corpus.items(), desc=f"[{self.cfg.dataset_name}] write JsonCollection"):
                title = (rec.get("title") or "").strip()
                text  = (rec.get("text") or rec.get("contents") or "").strip()
                contents = (title + "\n\n" + text).strip() if title else text
                f.write(json.dumps({"id": cid, "contents": contents}, ensure_ascii=False) + "\n")
        print(f"[convert] wrote {out_path}")

    # ---------- index ----------
    def _build_index(self):
        if self.index_dir.exists() and any(self.index_dir.iterdir()):
            print(f"[index] exists: {self.index_dir}"); return
        cmd = [
            os.path.basename(os.sys.executable), "-m", "pyserini.index.lucene",
            "--collection", "JsonCollection",
            "--input", str(self.jsonl_dir),
            "--index", str(self.index_dir),
            "--generator", "DefaultLuceneDocumentGenerator",
            "--threads", str(self.cfg.index_threads),
            "--storePositions", "--storeDocvectors", "--storeRaw", "--optimize"
        ]
        print(f"[index] building at {self.index_dir} …")
        subprocess.run(cmd, check=True)
        print("[index] done.")

    def _load_query_answers(self) -> Dict[str, Any]:
        """
        Load gold answers from the queries jsonl file when available.
        For BEIR datasets, answers are in metadata.answer.
        """
        answers: Dict[str, Any] = {}

        if self.queries_file is None or not self.queries_file.exists():
            return answers

        if self.queries_file.suffix.lower() not in {".jsonl", ".json"}:
            return answers

        with open(self.queries_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                qid = str(obj.get("_id") or obj.get("id") or "")
                if not qid:
                    continue

                meta = obj.get("metadata", {})
                ans = None

                if isinstance(meta, dict) and "answer" in meta:
                    ans = meta["answer"]

                # Fallbacks: if other datasets use different fields
                if ans is None and "answer" in obj:
                    ans = obj["answer"]
                if ans is None and "answers" in obj:
                    ans = obj["answers"]

                if ans is not None:
                    answers[qid] = {'text': obj.get('text'), 'answer': ans}

        return answers

    def _make_searcher(self) -> LuceneSearcher:
        s = LuceneSearcher(str(self.index_dir))
        s.set_bm25(k1=self.cfg.k1, b=self.cfg.b)
        return s

    @staticmethod
    def _raw_to_text(raw: str) -> str:
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict) and "contents" in obj:
                return obj["contents"]
        except Exception:
            pass
        return raw

    def retrieve(self, queries: Dict[str, str]) -> Dict[str, Dict[str, float]]:
        qids = list(queries.keys())
        qtexts = [queries[qid] for qid in qids]
        s = self._make_searcher()
        hits = s.batch_search(queries=qtexts, qids=qids, k=self.cfg.k, threads=self.cfg.threads)
        return {qid: {h.docid: h.score for h in hits[qid]} for qid in hits.keys()}

    def eval_trec(self, qrels: Optional[Dict[str, Dict[str, int]]],
                  run: Dict[str, Dict[str, float]]) -> Optional[Dict[str, float]]:
        if qrels is None:
            print("[eval] no qrels; skip."); return None
        if pytrec_eval is None:
            print("[eval] pytrec_eval not installed; skip."); return None
        evaluator = pytrec_eval.RelevanceEvaluator(
            qrels,
            {'P_1','P_3','P_5','P_10','recall.3','recall.5','recall.10','map_cut_10','ndcg_cut_10'}
        )
        res = evaluator.evaluate(run)
        df = pd.DataFrame(res).T
        avg = df.mean().to_dict()
        print("[eval] " + ", ".join(f"{k}={v:.4f}" for k, v in avg.items()))
        return avg

    def write_trec_run(self, run: Dict[str, Dict[str, float]]) -> Path:
        out = self.out_dir / f"{self.cfg.dataset_name}__{self.cfg.split}__{self.cfg.run_tag}.trecrun"
        with open(out, "w", encoding="utf-8") as f:
            for qid, docmap in run.items():
                rows = sorted(docmap.items(), key=lambda kv: kv[1], reverse=True)
                for rank, (docid, score) in enumerate(rows, 1):
                    f.write(f"{qid} Q0 {docid} {rank} {score:.6f} {self.cfg.run_tag}\n")
        print(f"[run] wrote {out}")
        return out

    def write_llm_cache(self, run: Dict[str, Dict[str, float]], queries: Dict[str, str], answers: Dict[str, (str, str)]) -> Path:
        out = self.out_dir / f"{self.cfg.dataset_name}__{self.cfg.split}__{self.cfg.run_tag}__llm_cache.json"
        s = self._make_searcher()
        saved = []
        for qid, docmap in run.items():
            rows = sorted(docmap.items(), key=lambda kv: kv[1], reverse=True)
            ctxs = []
            for docid, score in rows:
                item = {"id": docid, "bm25_score": float(score)}
                if self.cfg.include_text_in_cache:
                    d = s.doc(docid)
                    if d is not None:
                        item["text"] = self._raw_to_text(d.raw())
                ctxs.append(item)
            saved.append({"id": qid, "input": queries[qid], "answers": answers[qid]['answer'], "ctxs": ctxs})
        with open(out, "w", encoding="utf-8") as f:
            json.dump(saved, f, ensure_ascii=False, indent=2)
        print(f"[cache] wrote {out}")
        return out

    def run_all(self) -> Tuple[Dict[str, str], Dict[str, Dict[str, float]]]:
        data_path = self.download_dataset()
        self._locate_files(data_path)

        query_answers = self._load_query_answers()

        corpus, queries, qrels = GenericDataLoader(
            corpus_file=str(self.corpus_file),
            query_file=str(self.queries_file),
            qrels_file=str(self.qrels_file) if self.qrels_file else None
        ).load_custom()

        self._write_json_for_pyserini(corpus)
        self._build_index()

        run = self.retrieve(queries)
        self.eval_trec(qrels, run)

        self.write_trec_run(run)
        self.write_llm_cache(run, queries, answers=query_answers)

        stats = LuceneIndexReader(str(self.index_dir)).stats()
        print(f"[stats] docs={stats['documents']:,} unique_terms={stats['unique_terms']:,} total_terms={stats['total_terms']:,}")
        return queries, run


def run_beir_bm25(
    datasets: List[str],
    split: str = "test",
    work_root: str | Path = "work",
    k: int = 100,
    threads: int = 32,
    k1: float = 0.9,
    b: float = 0.4,
    run_tag: str = "bm25",
    index_threads: int = 16,
    include_text_in_cache: bool = True,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Run BM25 for one or more BEIR datasets programmatically.

    Returns:
      results: dict mapping dataset_name -> {"queries": <dict[qid->text]>, "run": <dict[qid->{docid:score}>}
    """
    results = {}
    work_root = Path(work_root)
    for ds in datasets:
        print(f"\n=== Dataset: {ds} | Split: {split} ===")
        cfg = BeirBM25Config(
            dataset_name=ds,
            split=split,
            work_root=work_root,
            k=k, threads=threads, k1=k1, b=b,
            run_tag=run_tag,
            index_threads=index_threads,
            include_text_in_cache=include_text_in_cache
        )
        runner = BeirBM25Runner(cfg)
        queries, run = runner.run_all()
        results[ds] = {"queries": queries, "run": run}

    return results


class Evaluator:
    def __init__(self,
                 cache_path: str | Path,
                 model_name: str,
                 dataset_name: str,
                 max_new_tokens: int = 32,
                 max_length: int = 1200,
                 top_k_docs: int = 20,
                 n_std: float = 0.5,
                 device: Optional[str] = None,
                 n_judge_plots=10):
        self.cache_path = Path(cache_path)
        self.model_name = model_name
        self.dataset_name = dataset_name

        self.max_length = max_length
        self.max_new_tokens = max_new_tokens

        self.top_k_docs = top_k_docs
        self.n_std = n_std
        self.device = self._resolve_device(device)

        self.entries = self._load_cache()
        self.tokenizer, self.model = self._load_model()

        yes_ids = self.tokenizer.encode(" Yes", add_special_tokens=False)
        no_ids = self.tokenizer.encode(" No", add_special_tokens=False)
        assert len(yes_ids) == 1 and len(no_ids) == 1, "Expected single-token ' Yes'/' No'."
        self.yes_id, self.no_id = yes_ids[0], no_ids[0]

        self.generation_config = self._build_generation_config()

        saving_directory = Path('Results')
        saving_directory.mkdir(exist_ok=True)
        self.saving_directory = saving_directory

        self.n_judge_plots = n_judge_plots


    def _build_generation_config(self) -> GenerationConfig:
        try:
            gen_cfg = GenerationConfig.from_pretrained(self.model_name)
        except Exception:
            gen_cfg = GenerationConfig()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if gen_cfg.pad_token_id is None:
            gen_cfg.pad_token_id = self.tokenizer.pad_token_id
        if gen_cfg.eos_token_id is None:
            gen_cfg.eos_token_id = self.tokenizer.eos_token_id

        # override core things you care about
        gen_cfg.max_new_tokens = self.max_new_tokens
        gen_cfg.do_sample = False
        gen_cfg.temperature = 0.0
        gen_cfg.repetition_penalty = 1.5

        return gen_cfg

    def _load_cache(self) -> List[Dict[str, Any]]:
        with open(self.cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def _resolve_device(self, device: Optional[str]) -> str:
        if device is not None:
            return device
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _normalize_answer(self, answer):
        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(answer))))

    def _compute_accuracy(self, preds: List[str], gold_list: List[List[str]]) -> float:
        exact_match = 0
        total = 0
        for pred, golds in zip(preds, gold_list):
            if not golds:
                continue
            total += 1
            p = self._normalize_answer(pred)
            if any(self._normalize_answer(g) == p for g in golds):
                exact_match += 1
        exact_match_accuracy = exact_match / total if total > 0 else 0.0
        return exact_match_accuracy

    def _load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            device_map="auto" if self.device != "cpu" else None,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if model.config.pad_token_id is None:
            model.config.pad_token_id = model.config.eos_token_id

        model.to(self.device)
        model.eval()
        return tokenizer, model

    def _truncate_to_max_tokens(self, text: str) -> str:
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )
        ids = tokens["input_ids"]
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def _build_qa_prompt(self, query: str, docs: List[str]) -> str:
        context = "\n\n".join(docs)
        context = self._truncate_to_max_tokens(context)
        prompt = (
            "Answer the Question using only the provided Context passages.\n"
            "Your response must be AFTER the the Answer:\n"
            "Output ONLY the final answer, with no explanation, no extra words, no quotes, and no punctuation before or after.\n"
            "Example 1:\n"
            "Context:\n"
            "\"1984 is a dystopian social science fiction novel and cautionary tale written by George Orwell.\"\n"
            "Question: Who wrote the novel 1984?\n"
            "Answer: \nGeorge Orwell\n\n"
            "Example 2:\n"
            "Context:\n"
            "\"Paris is the capital and most populous city of France.\"\n"
            "Question: What is the capital of France?\n"
            "Answer: \nParis\n\n"
            "Now answer the next question using the Context.\n\n"
            f"Question: {query}\n"
            f"Context:\n{context}\n"
            f"Answer:\n"
        )
        return prompt

    def _build_judge_prompt(self, query: str, doc: str, answer: str) -> str:
        prompt = (
            "You are a noisy document evaluator that can judge if the external document is noisy for the query with "
            "unrelated or misleading information. Given a retrieved Document, a Question, and an Answer generated by "
            "an LLM (LLM Answer), you should judge whether both the following two conditions are reached: (1) the "
            "Document provides specific information for answering the Question; (2) the LLM Answer directly answers "
            "the question based on the retrieved Document. Please note that external documents may contain noisy or "
            "factually incorrect information. If the information in the document does not contain the answer, you should "
            "point it out with evidence. You should answer with ' Yes' or ' No' with evidence of your judgment, "
            "where ' No' means one of the conditions (1) and (2) are unreached and indicates it is a noisy document."
            "Your verdict judgment outcome must be only be one of two options ' Yes' or ' No'."
            f"Question: {query}\n\n"
            f"Document:\n{doc}\n\n"
            f"LLM Answer: {answer}\n\n"
        )
        return prompt

    def _build_contextless_prompt(self, query):
        prompt = (
            "Answer the Question using only your knowledge.\n"
            "Your response must be AFTER the the Answer:\n"
            "Output ONLY the final answer, with no explanation, no extra words, no quotes, and no punctuation before or after.\n"
            "Example 1:\n"
            "Question: Who wrote the novel 1984?\n"
            "Answer: \nGeorge Orwell\n\n"
            "Example 2:\n"
            "Question: What is the capital of France?\n"
            "Answer: \nParis\n\n"
            "Now answer the next question using your knowledge.\n\n"
            f"Question: {query}\n"
            f"Answer:\n"
        )
        return prompt

    @torch.no_grad()
    def _judge_single(self, prompt: str) -> float:
        inputs = self.tokenizer(
            prompt,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        logits = outputs.logits[:, -1, :]  # n_doc * vocab_size
        log_probs = torch.log_softmax(logits, dim=-1)
        yes_logp = log_probs[:, self.yes_id]
        no_logp = log_probs[:, self.no_id]
        score = (yes_logp - no_logp).cpu().tolist()[0]
        return score

    @torch.no_grad()
    def _generate_single(self, prompt):
        inputs = self.tokenizer([prompt], padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_length = inputs["attention_mask"].sum(dim=1)

        output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                     generation_config=self.generation_config, )

        # text = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        # prediction = text[len(prompt):].split("\n")[0]

        gen_ids = output[:, input_length:].tolist() # only the generated continuation tokens
        # whole_text = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        gen_text = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]

        # print(whole_text)

        return gen_text

    def run_vanilla_rag(self, entries: List[Dict[str, Any]]) -> List[str]:

        entries_prompts = []
        for entry in entries:
            query = entry["input"]
            ctxs = sorted(entry["ctxs"], key=lambda x: x["bm25_score"], reverse=True)
            ctxs = ctxs[: self.top_k_docs]
            docs = [self._truncate_to_max_tokens(c["text"]) for c in ctxs]

            prompt = self._build_qa_prompt(query, docs)
            entries_prompts.append(prompt)

        predictions = [self._generate_single(prompt) for prompt in entries_prompts]

        return predictions

    def plot_threshold(self, scores, threshold, method, qid, bins=10):
        plt.figure(figsize=(3, 2.5))
        plt.hist(scores, bins=bins, edgecolor="black")
        plt.axvline(threshold, linestyle="--", linewidth=1.5, color='red')
        plt.xlabel("Judge Score")
        plt.ylabel("Count")
        plt.tight_layout()
        save_directory = self.saving_directory / 'Judges'
        save_directory.mkdir(exist_ok=True, parents=True)
        filename = qid + "_" + method + "_" + self.dataset_name  +  "_" +  self.model_name.split("/")[0] +'.png'
        plt.savefig(save_directory / filename, dpi=300)
        plt.close()

    def run_main_rag(self, entries: List[Dict[str, Any]]) -> Tuple[List[str], List[str], List[str]]:
        # predictions: List[str] = []
        predictions_mainrag: List[str] = []
        predictions_quantile: List[str] = []
        predictions_otsu: List[str] = []

        for entry in entries:
            query = entry["input"]
            ctxs = sorted(entry["ctxs"], key=lambda x: x["bm25_score"], reverse=True)
            ctxs = ctxs[: self.top_k_docs]

            # --- Stage 1: Predictor per doc (batched when possible) ---
            docs = [self._truncate_to_max_tokens(c["text"]) for c in ctxs]
            predictor_prompts = [self._build_qa_prompt(query, [doc]) for doc in docs]
            pred_answers = [self._generate_single(prompt) for prompt in predictor_prompts]

            doc_answers: List[Dict[str, Any]] = []
            for ctx, doc, ans in zip(ctxs, docs, pred_answers):
                doc_answers.append(
                    {
                        "doc_id": ctx["id"],
                        "doc_text": doc,
                        "predicted_answer": ans,
                    }
                )

            # --- Stage 2: Judge per (Q, D, A) ---
            judge_prompts = [self._build_judge_prompt(query, da["doc_text"], da["predicted_answer"])
                             for da in doc_answers]

            scores = [self._judge_single(prompt) for prompt in judge_prompts]

            for da, s in zip(doc_answers, scores):
                da["judge_score"] = s

            # --- Stage 3: thresholding ---
            finite_scores = [s for s in scores if math.isfinite(s)]
            # if finite_scores:
            #     mu = statistics.mean(finite_scores)
            #     sigma = statistics.pstdev(finite_scores) if len(finite_scores) > 1 else 0.0
            #     threshold = mu - self.n_std * sigma
            #
            #     kept = [da for da in doc_answers if da["judge_score"] >= threshold]
            #     kept_sorted = sorted(kept, key=lambda x: x["judge_score"], reverse=True)
            #
            #     # If everything got filtered out, keep highest scored doc
            #     if not kept_sorted and doc_answers:
            #         kept_sorted = sorted(doc_answers, key=lambda x: x["judge_score"], reverse=True)[:1]
            #
            #     docs_for_final = [x["doc_text"] for x in kept_sorted]
            # else:
            #     # degenerate case: fall back to all docs
            #     docs_for_final = docs

            n_docs = len(finite_scores)

            # --- thresholds ---
            mu = statistics.mean(finite_scores)
            sigma = statistics.pstdev(finite_scores) if n_docs > 1 else 0.0
            main_rag_threshold = mu - self.n_std * sigma

            scores_sorted = sorted(finite_scores)
            q = getattr(self, "quantile_q", 0.8)
            quantile_threshold = scores_sorted[int(q * (n_docs - 1))]

            # Otsu-style threshold on sorted scores
            total, cumsum = sum(scores_sorted), 0.0
            best_i, best_var = 0, -1.0
            for i, s in enumerate(scores_sorted[:-1], 1):
                cumsum += s
                w0, w1 = i / n_docs, 1 - i / n_docs
                mu0, mu1 = cumsum / i, (total - cumsum) / (n_docs - i)
                var_between = w0 * w1 * (mu0 - mu1) ** 2
                if var_between > best_var:
                    best_var, best_i = var_between, i - 1
            otsu_threshold = scores_sorted[best_i]

            def select(th):
                kept = [da for da in doc_answers if da["judge_score"] >= th]
                if not kept:
                    kept = [max(doc_answers, key=lambda x: x["judge_score"])]
                return [da["doc_text"] for da in sorted(kept, key=lambda x: x["judge_score"], reverse=True)]

            docs_for_final_mainrag = select(main_rag_threshold)
            docs_for_final_quantile = select(quantile_threshold)
            docs_for_final_otsu = select(otsu_threshold)

            if self.n_judge_plots > 0:
                self.plot_threshold(finite_scores, main_rag_threshold, 'mainrag', entry['id'][:6])
                self.plot_threshold(finite_scores, quantile_threshold, 'quantile', entry['id'][:6])
                self.plot_threshold(finite_scores, otsu_threshold, 'otsu', entry['id'][:6])
                self.n_judge_plots -= 1

            # --- Stage 4: Final RAG answer (single prompt) ---
            final_prompt_mainrag = self._build_qa_prompt(query, docs_for_final_mainrag)
            final_prompt_quantile = self._build_qa_prompt(query, docs_for_final_quantile)
            final_prompt_otsu = self._build_qa_prompt(query, docs_for_final_otsu)

            final_answer_mainrag = self._generate_single(final_prompt_mainrag)
            final_answer_quantile = self._generate_single(final_prompt_quantile)
            final_answer_otsu = self._generate_single(final_prompt_otsu)

            predictions_mainrag.append(final_answer_mainrag)
            predictions_quantile.append(final_answer_quantile)
            predictions_otsu.append(final_answer_otsu)

        return predictions_mainrag, predictions_quantile, predictions_otsu

    def run_llm_only_generations(self, entries: List[Dict[str, Any]]) -> List[str]:
        contextless_prompts = [self._build_contextless_prompt(entry["input"]) for entry in entries]
        predictions = [self._generate_single(prompt) for prompt in contextless_prompts]

        return predictions


    def run(self, max_queries: Optional[int] = None):
        entries = self.entries
        if max_queries is not None:
            rng = random.Random(42)
            rng.shuffle(entries)
            entries = entries[:max_queries]

        gold_list: List[List[str]] = [entry.get("answers") for entry in entries]

        # llm only without context
        llm_only_preds = self.run_llm_only_generations(entries)
        llm_only_accuracy = self._compute_accuracy(llm_only_preds, gold_list)

        # Vanilla RAG
        vanilla_preds = self.run_vanilla_rag(entries)
        vanilla_accuracy = self._compute_accuracy(vanilla_preds, gold_list)

        # MAIN-RAG
        # main_preds = self.run_main_rag(entries)
        predictions_mainrag, predictions_quantile, predictions_otsu = self.run_main_rag(entries)
        mainrag_accuracy = self._compute_accuracy(predictions_mainrag, gold_list)
        quantile_accuracy = self._compute_accuracy(predictions_quantile, gold_list)
        otsu_accuracy = self._compute_accuracy(predictions_otsu, gold_list)

        results = {
            "llm_only": {"accuracy": llm_only_accuracy},
            "vanilla": {"accuracy": vanilla_accuracy},
            "mainrag": {"accuracy": mainrag_accuracy},
            "mainrag_quantile": {"accuracy": quantile_accuracy},
            "mainrag_otsu": {"accuracy": otsu_accuracy},
        }

        self.save_results(results)
        self.n_judge_plots = 10

    def save_results(self, results: Dict[str, Dict[str, float]]):
        rows = []
        for pipeline_name, metrics in results.items():
            for metric_name, value in metrics.items():
                rows.append(
                    {
                        "model": self.model_name,
                        "dataset": self.dataset_name,
                        "pipeline": pipeline_name,
                        "metric": metric_name,
                        "value": value,
                    }
                )

        results_df = pd.DataFrame(rows)
        results_df.to_csv(self.saving_directory / f'{self.dataset_name}_{self.model_name.split("/")[0]}.csv', index=False)
        print(f'results with {self.model_name} for dataset {self.dataset_name}', results_df)


if __name__ == "__main__":
    # Example: BM25 cache path for hotpotqa with run_tag="bm25_base"
    dataset_names = ["hotpotqa"]  # add more: ["hotpotqa", "nq", "fiqa", "fever"]
    run_beir_bm25(dataset_names, split="test", k=50, threads=32, k1=0.9, b=0.4, run_tag="bm25_base")

    login(os.environ["HF_TOKEN"])
    # model_names = ["mistralai/Mistral-7B-Instruct-v0.3", 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    #                'google/gemma-2-27b-it', "Qwen/Qwen2.5-7B-Instruct"]
    model_names = ['google/gemma-2-27b-it']
    iteration_seeds = [42, 17, 1142, 117]

    print('BM25 is done!')

    max_queries = 50
    for model_name in model_names:
        for dataset_name in dataset_names:
            start_time = time.time()
            cache_path = f"work/{dataset_name}/runs/{dataset_name}__test__bm25_base__llm_cache.json"

            evaluator = Evaluator(
                cache_path=cache_path,
                model_name=model_name,
                dataset_name=dataset_name,
                max_new_tokens=32,
                top_k_docs=20,
                n_std=1.5,
                max_length=10000
            )

            evaluator.run(max_queries=max_queries)

            end_time = time.time()

            print(f'Run time duration via {model_name} for dataset {dataset_name} is :', round(end_time - start_time))