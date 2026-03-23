import sys
import json
from pathlib import Path

# Add parent directory to path so we can import from main
sys.path.append(str(Path(__file__).parent.parent))

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from main import retriever, format_docs_with_sources, generation_chain, finalize_response_with_sources, llm
from eval_dataset import RESPONSE_EVAL_SET

EVALUATOR_PROMPT = """Je bent een strenge, objectieve beoordelaar van AI-systemen.
Beoordeel het onderstaande AI-antwoord op basis van de vraag, de verwachte waarheid (ground truth), en de verstrekte context.

Geef je beoordeling als een JSON-object met exact deze sleutels:
- "faithfulness" (boolean): Is de stelling in het "AI Antwoord" gebaseerd op de "Context"? (False als de AI verzint/hallucineert).
- "correctness" (boolean): Komt de inhoud van het "AI Antwoord" overeen met de "Verwachte Waarheid"?
- "citation_accuracy" (boolean): Zijn de inline bronverwijzingen (zoals [Bron 1], [Bron 2]) correct geplaatst bij de juiste informatie? Komen de claims bij een bepaalde [Bron X] daadwerkelijk uit die specifieke bron in de verstrekte context? (Beoordeel met True als er geen bron in de antwoord zit maar dit terecht is).
- "reasoning" (string): Korte uitleg (max 3 zinnen) van je beoordeling.

Vraag: {question}
Verwachte Waarheid: {ground_truth}

Context (De enige informatie die de AI mocht gebruiken, let goed op de [Bron X] labels):
{context}

---
AI Antwoord:
{answer}
"""

eval_prompt = ChatPromptTemplate.from_template(EVALUATOR_PROMPT)
eval_chain = eval_prompt | llm | JsonOutputParser()

def test_responses():
    print("="*60)
    print("STARTING RESPONSE EVALUATION")
    print("="*60)
    
    total_tests = len(RESPONSE_EVAL_SET)
    faithfulness_score = 0
    correctness_score = 0
    citation_accuracy_score = 0
    
    results = []

    for entry in RESPONSE_EVAL_SET:
        print(f"\nEvaluating: [{entry['id']}] - {entry['type']}")
        
        # 1. Run the RAG pipeline to get the context and the answer
        docs = retriever.invoke(entry["question"])
        context = format_docs_with_sources(docs)
        
        raw_answer = generation_chain.invoke({
            "context": context, 
            "history": "",  # Added missing history variable for tests
            "question": entry["question"]
        })
        final_answer = finalize_response_with_sources(raw_answer, docs)
        
        # 2. Evaluate the answer using the LLM judge
        try:
            evaluation = eval_chain.invoke({
                "question": entry["question"],
                "ground_truth": entry["ground_truth"],
                "context": context,
                "answer": final_answer
            })
        except Exception as e:
            print(f"  [Error] Failed to parse evaluator JSON: {e}")
            evaluation = {"faithfulness": False, "correctness": False, "reasoning": "Parse error"}

        # 3. Print test results
        is_faithful = evaluation.get('faithfulness', False)
        is_correct = evaluation.get('correctness', False)
        is_citation_accurate = evaluation.get('citation_accuracy', False)
        
        faithfulness_score += 1 if is_faithful else 0
        correctness_score += 1 if is_correct else 0
        citation_accuracy_score += 1 if is_citation_accurate else 0
        
        print(f"  Faithfulness (No Hallucinations): {'✅ Pass' if is_faithful else '❌ Fail'}")
        print(f"  Correctness (Accurate Info):      {'✅ Pass' if is_correct else '❌ Fail'}")
        print(f"  Citation Accuracy: {'✅ Pass' if is_citation_accurate else '❌ Fail'}")
        print(f"  Reasoning: {evaluation.get('reasoning', '')}")
        
        # Save exact context and response for debugging
        results.append({
            "id": entry["id"],
            "question": entry["question"],
            "ai_response": final_answer,
            "evaluation": evaluation
        })

    # Print summary report
    print("\n" + "="*60)
    print("EVALUATION REPORT")
    print("="*60)
    print(f"Total Tests:  {total_tests}")
    print(f"Faithfulness: {faithfulness_score}/{total_tests} ({(faithfulness_score/total_tests)*100:.1f}%)")
    print(f"Correctness:  {correctness_score}/{total_tests} ({(correctness_score/total_tests)*100:.1f}%)")
    print(f"Citation Accuracy: {citation_accuracy_score}/{total_tests} ({(citation_accuracy_score/total_tests)*100:.1f}%)")
    print("="*60)
    
    # Optionally dump to file
    with open("response_eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print("Detailed results saved to response_eval_results.json")

if __name__ == "__main__":
    test_responses()
