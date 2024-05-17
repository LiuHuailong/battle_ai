import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import random

# GPT-2モデルとトークナイザの読み込み
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def evaluate_skill_strength(keyword):
    # キーワードをトークン化
    input_ids = tokenizer.encode(keyword, return_tensors="pt")

    # モデルに入力してスキルを生成
    with torch.no_grad():
        output = model.generate(input_ids, max_length=20, num_return_sequences=1)
        generated_skill = tokenizer.decode(output[0], skip_special_tokens=True)

    # スキルの強さをモデルの出力確率から抽出
    skill_strength = output[0].max().item()  # 0から1の範囲でスコアを取得

    return generated_skill, skill_strength

def generate_keyword():
    # AIがキーワードを生成
    keywords = ["AI", "Python", "GPT-2", "対戦", "スキル"]
    return random.choice(keywords)

def main():
    # 初回のキーワードでHPを生成
    user_keyword = input("初回のキーワードを入力してください: ")
    _, user_hp = evaluate_skill_strength(user_keyword)
    user_hp *= 20  # スキルの強さをHPに変換

    ai_keyword = generate_keyword()
    _, ai_hp = evaluate_skill_strength(ai_keyword)
    ai_hp *= 20  # スキルの強さをHPに変換
    
    print(f"あなたの生成されたHPコマンド: {user_keyword}")
    print(f"あなたの生成されたHP: {user_hp}")
    print(f"AIのHPコマンド: {ai_keyword}")
    print(f"AIのHP: {ai_hp}")

    while True:
        # ユーザーのターン
        keyword = input("キーワードを入力してください (qで終了): ")
        if keyword.lower() == "q":
            break

        skill, strength = evaluate_skill_strength(keyword)
        ai_hp -= strength * 10  # スキルの強さに基づいてHPを減らす
        print(f"あなたの生成されたスキル: {skill}")
        print(f"あなたのスキルの強さ: {strength:.2f}")
        print(f"AIの残りHP: {ai_hp:.2f}")

        if ai_hp <= 0:
            print("あなたの勝利です！")
            break

        # AIのターン
        keyword = generate_keyword()
        skill, strength = evaluate_skill_strength(keyword)
        user_hp -= strength * 10  # スキルの強さに基づいてHPを減らす
        print(f"AIの生成されたスキル: {skill}")
        print(f"AIのスキルの強さ: {strength:.2f}")
        print(f"あなたの残りHP: {user_hp:.2f}")

        if user_hp <= 0:
            print("AIの勝利です！")
            break

if __name__ == "__main__":
    main()
