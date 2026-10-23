"""
Automated Multimodal API Test Cases Script
Runs a full test suite against http://localhost:8000/predict and http://localhost:8000/explain_multimodal
"""

import requests
import json
import os

API_URL = "http://localhost:8000/predict"
EXPLAIN_URL = "http://localhost:8000/explain_multimodal"

TEST_CASES = [
    # ----------------------------------------------------
    # Category 1: Multilingual Text Test Cases (10+ Languages)
    # ----------------------------------------------------
    {
        "id": "TC-TEXT-01",
        "category": "Multilingual Text (English - Positive)",
        "text": "This product is absolutely amazing! The build quality exceeded all my expectations.",
        "expected_sentiment": "POSITIVE"
    },
    {
        "id": "TC-TEXT-02",
        "category": "Multilingual Text (English - Negative)",
        "text": "Terrible customer service and extremely poor product durability. Complete waste of money.",
        "expected_sentiment": "NEGATIVE"
    },
    {
        "id": "TC-TEXT-03",
        "category": "Multilingual Text (Spanish - Positive)",
        "text": "¡Me encanta este servicio! Es realmente maravilloso y muy eficiente.",
        "expected_sentiment": "POSITIVE"
    },
    {
        "id": "TC-TEXT-04",
        "category": "Multilingual Text (French - Negative)",
        "text": "Très mauvaise expérience, la commande est arrivée en retard et le produit est défectueux.",
        "expected_sentiment": "NEGATIVE"
    },
    {
        "id": "TC-TEXT-05",
        "category": "Multilingual Text (German - Positive)",
        "text": "Ich bin absolut begeistert von dieser hervorragenden Qualität und dem tollen Support!",
        "expected_sentiment": "POSITIVE"
    },
    {
        "id": "TC-TEXT-06",
        "category": "Multilingual Text (Hindi - Positive)",
        "text": "यह अनुभव बहुत ही शानदार और बेहतरीन रहा! मुझे बहुत खुशी हुई।",
        "expected_sentiment": "POSITIVE"
    },
    {
        "id": "TC-TEXT-07",
        "category": "Multilingual Text (Tamil - Negative)",
        "text": "மிகவும் மோசமான சேவை, பண விரயம் மட்டுமே கிடைத்தது.",
        "expected_sentiment": "NEGATIVE"
    },
    {
        "id": "TC-TEXT-08",
        "category": "Multilingual Text (Chinese - Positive)",
        "text": "强烈推荐这个产品！质量非常出众，售后服务也特别好。",
        "expected_sentiment": "POSITIVE"
    },
    {
        "id": "TC-TEXT-09",
        "category": "Multilingual Text (Japanese - Negative)",
        "text": "使いにくくて品質がひどいです。二度と購入しません。",
        "expected_sentiment": "NEGATIVE"
    },
    {
        "id": "TC-TEXT-10",
        "category": "Multilingual Text (Arabic - Neutral)",
        "text": "وصل المنتج اليوم في موعده والتغليف عالي الجودة بشكل عادي.",
        "expected_sentiment": "NEUTRAL"
    },
    {
        "id": "TC-TEXT-11",
        "category": "Multilingual Text (Russian - Positive)",
        "text": "Отличный сервис, высококачественные материалы и быстрая доставка!",
        "expected_sentiment": "POSITIVE"
    },

    # ----------------------------------------------------
    # Category 2: Neutral / Factual Statements
    # ----------------------------------------------------
    {
        "id": "TC-TEXT-12",
        "category": "Factual Neutral Narrative",
        "text": "The meeting is scheduled for Thursday at 3:00 PM in Conference Room B.",
        "expected_sentiment": "NEUTRAL"
    }
]

import sys

def run_test_suite():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        
    print("==================================================")
    print("      AUTOMATED MULTIMODAL TEST SUITE BENCHMARK    ")
    print("==================================================")
    
    passed = 0
    failed = 0

    for tc in TEST_CASES:
        print(f"\nRunning {tc['id']} [{tc['category']}]:")
        print(f"  Input Text: \"{tc['text']}\"")
        
        try:
            res = requests.post(API_URL, data={"text": tc["text"]})
            if res.status_code == 200:
                data = res.json()
                pos, neu, neg = data["positive"], data["neutral"], data["negative"]
                
                # Determine predicted label
                max_val = max(pos, neu, neg)
                if max_val == pos:
                    pred_label = "POSITIVE"
                elif max_val == neg:
                    pred_label = "NEGATIVE"
                else:
                    pred_label = "NEUTRAL"

                is_pass = (pred_label == tc["expected_sentiment"])
                if is_pass:
                    passed += 1
                    status_str = "PASS [✓]"
                else:
                    failed += 1
                    status_str = "FAIL [✗]"

                print(f"  Predicted: {pred_label:<8} (Pos: {pos:.3f}, Neu: {neu:.3f}, Neg: {neg:.3f}) | Expected: {tc['expected_sentiment']} ==> {status_str}")
            else:
                print(f"  API Error: Status Code {res.status_code}")
                failed += 1
        except Exception as e:
            print(f"  Connection Error: {e}")
            failed += 1

    print("\n--------------------------------------------------")
    print(f"TEST SUITE SUMMARY:  Total: {len(TEST_CASES)} | Passed: {passed} | Failed: {failed}")
    print("--------------------------------------------------")

if __name__ == '__main__':
    run_test_suite()
