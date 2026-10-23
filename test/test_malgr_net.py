"""
MALGR-Net Test Suite Script
Executes PyTorch architecture unit tests + 12 multilingual integration test cases for MALGR-Net.
"""

import os
import sys
import requests
import torch

# Ensure project root is in path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.malgr_net import MALGRNet, MALGRLoss

# Fix Windows console UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

TEST_CASES = [
    {"id": "TC-TEXT-01", "desc": "Multilingual Text (English - Positive)", "text": "This product is absolutely amazing! The build quality exceeded all my expectations.", "expected": "POSITIVE"},
    {"id": "TC-TEXT-02", "desc": "Multilingual Text (English - Negative)", "text": "Terrible customer service and extremely poor product durability. Complete waste of money.", "expected": "NEGATIVE"},
    {"id": "TC-TEXT-03", "desc": "Multilingual Text (Spanish - Positive)", "text": "¡Me encanta este servicio! Es realmente maravilloso y muy eficiente.", "expected": "POSITIVE"},
    {"id": "TC-TEXT-04", "desc": "Multilingual Text (French - Negative)", "text": "Très mauvaise expérience, la commande est arrivée en retard et le produit est défectueux.", "expected": "NEGATIVE"},
    {"id": "TC-TEXT-05", "desc": "Multilingual Text (German - Positive)", "text": "Ich bin absolut begeistert von dieser hervorragenden Qualität und dem tollen Support!", "expected": "POSITIVE"},
    {"id": "TC-TEXT-06", "desc": "Multilingual Text (Hindi - Positive)", "text": "यह अनुभव बहुत ही शानदार और बेहतरीन रहा! मुझे बहुत खुशी हुई।", "expected": "POSITIVE"},
    {"id": "TC-TEXT-07", "desc": "Multilingual Text (Tamil - Negative)", "text": "மிகவும் மோசமான சேவை, பண விரயம் மட்டுமே கிடைத்தது.", "expected": "NEGATIVE"},
    {"id": "TC-TEXT-08", "desc": "Multilingual Text (Chinese - Positive)", "text": "强烈推荐这个产品！质量非常出众，售后服务也特别好。", "expected": "POSITIVE"},
    {"id": "TC-TEXT-09", "desc": "Multilingual Text (Japanese - Negative)", "text": "使いにくくて品質がひどいです。二度と購入しません。", "expected": "NEGATIVE"},
    {"id": "TC-TEXT-10", "desc": "Multilingual Text (Arabic - Neutral)", "text": "وصل المنتج اليوم في موعده والتغليف عالي الجودة بشكل عادي.", "expected": "NEUTRAL"},
    {"id": "TC-TEXT-11", "desc": "Multilingual Text (Russian - Positive)", "text": "Отличный сервис, высококачественные материалы и быстрая доставка!", "expected": "POSITIVE"},
    {"id": "TC-TEXT-12", "desc": "Factual Neutral Narrative", "text": "The meeting is scheduled for Thursday at 3:00 PM in Conference Room B.", "expected": "NEUTRAL"},
]

def test_malgr_net_architecture():
    print("--- Running MALGR-Net Architecture Unit Tests ---")
    model = MALGRNet(text_dim=768, audio_dim=16, image_dim=768, latent_dim=256, num_classes=7)
    
    t = torch.randn(4, 768)
    a = torch.randn(4, 16)
    v = torch.randn(4, 768)
    reliability = torch.tensor([[0.8, 0.2, 0.9], [0.1, 0.7, 0.2], [0.5, 0.5, 0.5], [0.9, 0.1, 0.1]])
    
    out = model(t, a, v, reliability=reliability)
    
    assert "logits" in out and "gates" in out and "zf" in out, "Missing required keys in MALGRNet output!"
    assert out["logits"].shape == (4, 7), f"Expected logits shape (4, 7), got {out['logits'].shape}"
    assert out["gates"].shape == (4, 3), f"Expected gates shape (4, 3), got {out['gates'].shape}"
    
    gate_sums = torch.sum(out["gates"], dim=-1)
    assert torch.allclose(gate_sums, torch.ones(4)), f"Gating weights do not sum to 1.0! Sums: {gate_sums}"
    
    loss_fn = MALGRLoss(lambda_c=0.05)
    targets = torch.tensor([3, 0, 4, 1])
    total_loss, l_cls, l_cons = loss_fn(out, targets)
    assert total_loss.item() > 0, "Loss computation failed!"
    
    print("[OK] MALGR-Net PyTorch Architecture Unit Tests Passed Cleanly!")

def run_tests():
    test_malgr_net_architecture()
    
    api_url = "http://localhost:8000/predict"
    print("\n==================================================")
    print("      AUTOMATED MALGR-NET TEST SUITE BENCHMARK    ")
    print("==================================================")
    
    passed = 0
    failed = 0
    
    for tc in TEST_CASES:
        print(f"\nRunning {tc['id']} [{tc['desc']}]:")
        print(f"  Input Text: \"{tc['text']}\"")
        try:
            res = requests.post(api_url, data={"text": tc['text']})
            if res.status_code == 200:
                data = res.json()
                pos, neu, neg = data["positive"], data["neutral"], data["negative"]
                gating = data.get("gating_weights", {})
                
                max_val = max(pos, neu, neg)
                if max_val == pos:
                    pred = "POSITIVE"
                elif max_val == neg:
                    pred = "NEGATIVE"
                else:
                    pred = "NEUTRAL"
                
                status = "PASS [✓]" if pred == tc['expected'] else "FAIL [✗]"
                if pred == tc['expected']:
                    passed += 1
                else:
                    failed += 1
                print(f"  Predicted: {pred:<8} (Pos: {pos:.3f}, Neu: {neu:.3f}, Neg: {neg:.3f}) | Gate [V:{gating.get('visual')}% A:{gating.get('audio')}% T:{gating.get('text')}%] ==> {status}")
            else:
                print(f"  API Error: HTTP {res.status_code}")
                failed += 1
        except Exception as e:
            print(f"  Connection Error: {e}")
            failed += 1

    print("\n--------------------------------------------------")
    print(f"TEST SUITE SUMMARY:  Total: {len(TEST_CASES)} | Passed: {passed} | Failed: {failed}")
    print("--------------------------------------------------")

if __name__ == "__main__":
    run_tests()
