# ============================================================
# FAITHFULNESS UTILS
# Tüm fonksiyonlar bu dosyada toplanmıştır
# OpenAI, Claude ve Gemini desteği
# ============================================================

import pandas as pd
import numpy as np
import json
import time
import os
from datetime import datetime
from scipy.spatial.distance import cosine

# --- Global Değişkenler (notebook'tan set edilecek) ---
embedding_model = None
nli_pipeline = None
openai_client = None
claude_client = None
gemini_model = None
VECTOR_WEIGHT = 0.4
NLI_WEIGHT = 0.6
LLM_MODEL = "gpt-4o-mini"
TEMPERATURE = 0
MAX_TOKENS = 2048
CHECKPOINT_PATH = ""


def set_models(emb_model, nli_pipe, openai_cli=None, claude_cli=None, gemini_mod=None):
    """Tüm modelleri set et"""
    global embedding_model, nli_pipeline, openai_client, claude_client, gemini_model
    embedding_model = emb_model
    nli_pipeline = nli_pipe
    openai_client = openai_cli
    claude_client = claude_cli
    gemini_model = gemini_mod


def set_config(vector_w, nli_w, llm_model, temp, max_tok, checkpoint_path):
    global VECTOR_WEIGHT, NLI_WEIGHT, LLM_MODEL, TEMPERATURE, MAX_TOKENS, CHECKPOINT_PATH
    VECTOR_WEIGHT = vector_w
    NLI_WEIGHT = nli_w
    LLM_MODEL = llm_model
    TEMPERATURE = temp
    MAX_TOKENS = max_tok
    CHECKPOINT_PATH = checkpoint_path


def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    print(f"✓ Veri yüklendi: {len(df)} satır")
    
    initial_count = len(df)
    df = df.dropna(subset=['gold_answer', 'gold_evidence', 'context', 'question'])
    
    dropped_count = initial_count - len(df)
    if dropped_count > 0:
        print(f"  ⚠️ {dropped_count} satır eksik veri nedeniyle çıkarıldı")
    
    df['gold_evidence_list'] = df['gold_evidence'].apply(
        lambda x: [item.strip() for item in str(x).split('|')]
    )
    df['evidence_count'] = df['gold_evidence_list'].apply(len)
    
    print(f"  Evidence parça sayıları: min={df['evidence_count'].min()}, "
          f"max={df['evidence_count'].max()}, "
          f"ortalama={df['evidence_count'].mean():.1f}")
    
    return df


def calculate_semantic_similarity(text1, text2):
    if not text1 or not text2:
        return 0.0
    
    embeddings = embedding_model.encode([text1, text2])
    similarity = 1 - cosine(embeddings[0], embeddings[1])
    
    return max(0.0, min(1.0, similarity))


def calculate_nli_score(premise, hypothesis):
    if not premise or not hypothesis:
        return {"label": "neutral", "scores": {}, "nli_score": 0.5}
    
    result = nli_pipeline(
        hypothesis,
        candidate_labels=["entailment", "neutral", "contradiction"],
        hypothesis_template="{}"
    )
    
    scores = dict(zip(result["labels"], result["scores"]))
    nli_score = (scores.get("entailment", 0) - scores.get("contradiction", 0) + 1) / 2
    
    return {
        "label": result["labels"][0],
        "scores": scores,
        "nli_score": max(0.0, min(1.0, nli_score))
    }


def calculate_faithfulness(reasoning, evidence_list):
    if not reasoning or not evidence_list:
        return {
            "semantic_similarity": 0.0,
            "nli_score": 0.5,
            "faithfulness_score": 0.25,
            "evidence_recall": 0.0,
            "details": []
        }
    
    details = []
    semantic_scores = []
    nli_scores = []
    evidence_used = 0
    
    for evidence in evidence_list:
        evidence = evidence.strip()
        if not evidence:
            continue
        
        sem_sim = calculate_semantic_similarity(reasoning, evidence)
        semantic_scores.append(sem_sim)
        
        nli_result = calculate_nli_score(evidence, reasoning)
        nli_scores.append(nli_result["nli_score"])
        
        if sem_sim > 0.5:
            evidence_used += 1
        
        details.append({
            "evidence": evidence[:50] + "..." if len(evidence) > 50 else evidence,
            "semantic_sim": round(sem_sim, 3),
            "nli_label": nli_result["label"],
            "nli_score": round(nli_result["nli_score"], 3)
        })
    
    avg_semantic = np.mean(semantic_scores) if semantic_scores else 0.0
    avg_nli = np.mean(nli_scores) if nli_scores else 0.5
    
    faithfulness = (VECTOR_WEIGHT * avg_semantic) + (NLI_WEIGHT * avg_nli)
    recall = evidence_used / len(evidence_list) if evidence_list else 0.0
    
    return {
        "semantic_similarity": round(avg_semantic, 4),
        "nli_score": round(avg_nli, 4),
        "faithfulness_score": round(faithfulness, 4),
        "evidence_recall": round(recall, 4),
        "details": details
    }


def calculate_accuracy(model_answer, gold_answer):
    if not model_answer or not gold_answer:
        return {"exact_match": False, "accuracy_score": 0}
    
    model_norm = str(model_answer).strip().lower()
    gold_norm = str(gold_answer).strip().lower()
    
    exact_match = model_norm == gold_norm
    
    return {
        "exact_match": exact_match,
        "accuracy_score": 1 if exact_match else 0
    }


def calculate_epistemic_metrics(confidence, faithfulness, accuracy):
    conf_norm = confidence / 100 if confidence else 0.5
    epistemic_gap = abs(conf_norm - faithfulness)
    
    if conf_norm > 0.7 and faithfulness < 0.5:
        hallucination_risk = "YÜKSEK"
    elif conf_norm > 0.7 and faithfulness < 0.7:
        hallucination_risk = "ORTA"
    else:
        hallucination_risk = "DÜŞÜK"
    
    if epistemic_gap < 0.15:
        calibration_type = "İyi Kalibre"
    elif conf_norm > faithfulness:
        calibration_type = "Overconfident"
    else:
        calibration_type = "Underconfident"
    
    return {
        "confidence_normalized": round(conf_norm, 4),
        "epistemic_gap": round(epistemic_gap, 4),
        "hallucination_risk": hallucination_risk,
        "calibration_type": calibration_type
    }


SYSTEM_PROMPT = """Sen bir problem çözme asistanısın. Sana verilen bağlam (context) ve soruyu (question) analiz edeceksin.

GÖREV:
1. Bağlamdaki bilgileri kullanarak adım adım düşün.
2. Her adımda hangi bilgiyi kullandığını açıkça belirt.
3. Sonuca ulaş ve cevabını ver.

ÇIKTI FORMATI (JSON):
{
    "reasoning": "Adım adım düşünce sürecin. Her adımda bağlamdan hangi bilgiyi kullandığını yaz.",
    "answer": "Nihai cevabın (kısa ve net)",
    "confidence": 0-100 arası bir sayı (cevabından ne kadar eminsin)
}

ÖNEMLİ:
- Sadece bağlamda verilen bilgileri kullan, dışarıdan bilgi ekleme.
- Cevabın SADECE geçerli bir JSON objesi olmalı, başka bir şey yazma.
- Türkçe yanıt ver."""


def extract_json_from_response(content):
    """Farklı formatlardan JSON çıkar"""
    # Doğrudan JSON ise
    try:
        return json.loads(content)
    except:
        pass
    
    # Markdown code block içinde
    if "```json" in content:
        try:
            json_str = content.split("```json")[1].split("```")[0].strip()
            return json.loads(json_str)
        except:
            pass
    
    if "```" in content:
        try:
            json_str = content.split("```")[1].split("```")[0].strip()
            return json.loads(json_str)
        except:
            pass
    
    # { ile başlayıp } ile biten kısmı bul
    try:
        start = content.find('{')
        end = content.rfind('}') + 1
        if start != -1 and end > start:
            json_str = content[start:end]
            return json.loads(json_str)
    except:
        pass
    
    return None


def generate_single_response(context, question):
    """Tek bir soru için cevap üret - tüm modeller desteklenir"""
    
    user_prompt = f"""BAĞLAM:
{context}

SORU:
{question}

Yukarıdaki bağlamı kullanarak soruyu cevapla. Yanıtını JSON formatında ver."""

    try:
        content = None
        
        # ========== CLAUDE ==========
        if LLM_MODEL.startswith("claude"):
            response = claude_client.messages.create(
                model=LLM_MODEL,
                max_tokens=MAX_TOKENS,
                messages=[
                    {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{user_prompt}"}
                ]
            )
            content = response.content[0].text
        
        # ========== GEMINI ==========
        elif LLM_MODEL.startswith("gemini"):
            full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
            response = gemini_model.generate_content(full_prompt)
            content = response.text
        
        # ========== OPENAI (varsayılan) ==========
        else:
            response = openai_client.chat.completions.create(
                model=LLM_MODEL,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
        
        # JSON parse et
        result = extract_json_from_response(content)
        
        if result is None:
            return {"error": "JSON parse edilemedi", "raw_response": content}
        
        # Gerekli alanları kontrol et
        required_fields = ['reasoning', 'answer', 'confidence']
        for field in required_fields:
            if field not in result:
                result[field] = None
        
        return result
        
    except Exception as e:
        return {"error": str(e)}


def generate_all_responses(df, checkpoint_name=None):
    from tqdm.notebook import tqdm
    
    results = []
    checkpoint_file = None
    processed_ids = set()
    
    if checkpoint_name:
        checkpoint_file = f"{CHECKPOINT_PATH}{checkpoint_name}.json"
        os.makedirs(CHECKPOINT_PATH, exist_ok=True)
        
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                saved_results = json.load(f)
                results = saved_results
                processed_ids = {r['question_id'] for r in saved_results}
                print(f"✓ Checkpoint yüklendi: {len(processed_ids)} soru zaten işlenmiş")
    
    to_process = df[~df['question_id'].isin(processed_ids)]
    
    if len(to_process) == 0:
        print("✓ Tüm sorular zaten işlenmiş!")
    else:
        print(f"► {len(to_process)} soru işlenecek...")
        print(f"  Model: {LLM_MODEL}")
        
        for idx, row in tqdm(to_process.iterrows(), total=len(to_process), desc="Generating"):
            response = generate_single_response(
                context=row['context'],
                question=row['question']
            )
            
            result_entry = {
                'question_id': row['question_id'],
                'model_reasoning': response.get('reasoning'),
                'model_answer': response.get('answer'),
                'model_confidence': response.get('confidence'),
                'error': response.get('error')
            }
            results.append(result_entry)
            
            if checkpoint_file and len(results) % 5 == 0:
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
            
            time.sleep(0.5)
        
        if checkpoint_file:
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"✓ Checkpoint kaydedildi: {checkpoint_file}")
    
    results_df = pd.DataFrame(results)
    merged_df = df.merge(results_df, on='question_id', how='left')
    
    success_count = merged_df['model_answer'].notna().sum()
    error_count = merged_df['error'].notna().sum()
    print(f"\n✓ Generation tamamlandı!")
    print(f"  Başarılı: {success_count}/{len(merged_df)}")
    if error_count > 0:
        print(f"  ⚠️ Hatalı: {error_count}")
    
    return merged_df


def run_evaluation(df_with_responses):
    from tqdm.notebook import tqdm
    
    print("► Evaluation başlıyor...")
    print(f"  Toplam soru: {len(df_with_responses)}")
    
    evaluation_results = []
    
    for idx, row in tqdm(df_with_responses.iterrows(), total=len(df_with_responses), desc="Evaluating"):
        
        faith_result = calculate_faithfulness(
            reasoning=row['model_reasoning'],
            evidence_list=row['gold_evidence_list']
        )
        
        acc_result = calculate_accuracy(
            model_answer=row['model_answer'],
            gold_answer=row['gold_answer']
        )
        
        epistemic_result = calculate_epistemic_metrics(
            confidence=row['model_confidence'],
            faithfulness=faith_result['faithfulness_score'],
            accuracy=acc_result['accuracy_score']
        )
        
        reasoning_length = len(str(row['model_reasoning'])) if row['model_reasoning'] else 0
        evidence_length = len(str(row['gold_evidence'])) if row['gold_evidence'] else 0
        
        evaluation_results.append({
            'question_id': row['question_id'],
            'semantic_similarity': faith_result['semantic_similarity'],
            'nli_score': faith_result['nli_score'],
            'faithfulness_score': faith_result['faithfulness_score'],
            'evidence_recall': faith_result['evidence_recall'],
            'exact_match': acc_result['exact_match'],
            'accuracy_score': acc_result['accuracy_score'],
            'confidence_normalized': epistemic_result['confidence_normalized'],
            'epistemic_gap': epistemic_result['epistemic_gap'],
            'hallucination_risk': epistemic_result['hallucination_risk'],
            'calibration_type': epistemic_result['calibration_type'],
            'reasoning_length': reasoning_length,
            'evidence_length': evidence_length,
            'length_ratio': reasoning_length / evidence_length if evidence_length > 0 else 0
        })
    
    eval_df = pd.DataFrame(evaluation_results)
    df_evaluated = df_with_responses.merge(eval_df, on='question_id', how='left')
    
    print("\n✓ Evaluation tamamlandı!")
    
    return df_evaluated


def print_evaluation_summary(df_evaluated):
    print("=" * 60)
    print("EVALUATION ÖZET SONUÇLARI")
    print("=" * 60)
    
    print("\n► ACCURACY")
    print(f"  Doğru cevap sayısı: {df_evaluated['accuracy_score'].sum()}/{len(df_evaluated)}")
    print(f"  Accuracy oranı: {df_evaluated['accuracy_score'].mean() * 100:.1f}%")
    
    print("\n► FAITHFULNESS")
    print(f"  Ortalama semantic similarity: {df_evaluated['semantic_similarity'].mean():.3f}")
    print(f"  Ortalama NLI score: {df_evaluated['nli_score'].mean():.3f}")
    print(f"  Ortalama faithfulness score: {df_evaluated['faithfulness_score'].mean():.3f}")
    print(f"  Ortalama evidence recall: {df_evaluated['evidence_recall'].mean():.3f}")
    
    print("\n► EPİSTEMİK ANALİZ")
    print(f"  Ortalama confidence: {df_evaluated['confidence_normalized'].mean() * 100:.1f}%")
    print(f"  Ortalama epistemic gap: {df_evaluated['epistemic_gap'].mean():.3f}")
    print(f"  Hallucination riski dağılımı:")
    for risk, count in df_evaluated['hallucination_risk'].value_counts().items():
        print(f"    {risk}: {count} ({count/len(df_evaluated)*100:.1f}%)")
    
    print(f"\n  Calibration tipi dağılımı:")
    for cal_type, count in df_evaluated['calibration_type'].value_counts().items():
        print(f"    {cal_type}: {count} ({count/len(df_evaluated)*100:.1f}%)")


def calculate_soft_accuracy(model_answer, gold_answer, semantic_threshold=0.85, nli_threshold=0.4):
    """
    Hibrit accuracy hesabı:
    1. Exact match varsa -> 1
    2. Semantic similarity >= threshold VE NLI entailment >= threshold -> 1
    3. Aksi halde -> 0
    """
    if not model_answer or not gold_answer:
        return {
            "exact_match": False,
            "soft_match": False,
            "accuracy_score": 0,
            "match_type": "no_answer",
            "semantic_sim": 0.0,
            "nli_entailment": 0.0
        }
    
    model_norm = str(model_answer).strip().lower()
    gold_norm = str(gold_answer).strip().lower()
    
    # 1. Exact match kontrolü
    if model_norm == gold_norm:
        return {
            "exact_match": True,
            "soft_match": True,
            "accuracy_score": 1,
            "match_type": "exact",
            "semantic_sim": 1.0,
            "nli_entailment": 1.0
        }
    
    # 2. Semantic similarity hesapla
    semantic_sim = calculate_semantic_similarity(model_answer, gold_answer)
    
    # 3. NLI entailment hesapla
    nli_result = nli_pipeline(
        model_answer,
        candidate_labels=["entailment", "neutral", "contradiction"],
        hypothesis_template="{}"
    )
    scores = dict(zip(nli_result["labels"], nli_result["scores"]))
    nli_entailment = scores.get("entailment", 0)
    
    # 4. Soft match kontrolü
    soft_match = (semantic_sim >= semantic_threshold) and (nli_entailment >= nli_threshold)
    
    return {
        "exact_match": False,
        "soft_match": soft_match,
        "accuracy_score": 1 if soft_match else 0,
        "match_type": "soft" if soft_match else "no_match",
        "semantic_sim": round(semantic_sim, 4),
        "nli_entailment": round(nli_entailment, 4)
    }
