import re
from fuzzywuzzy import process, fuzz
import matplotlib.pyplot as plt

# Negations
negation_words = {"not", "no", "never", "none", "nobody", "nothing", "neither", "nor", "nowhere", "n't"}

# Intensifiers
intensifiers = {
    "very": 2.0, "really": 2.0, "extremely": 2.5, "totally": 2.2, "absolutely": 2.2,
    "so": 1.8, "quite": 1.5, "pretty": 1.5, "somewhat": 0.7, "slightly": 0.6,
    "bit": 0.7, "kinda": 0.7, "kind": 0.7,
}

# Lexicon (add mais se quiser crescer)
lexicon = {
    "happy": (8.0, 5.9), "joyful": (8.3, 6.4), "delighted": (8.4, 6.8), "ecstatic": (8.7, 7.5),
    "thrilled": (8.1, 7.8), "excited": (7.8, 7.6), "enthusiastic": (7.9, 7.4), "glad": (7.8, 5.5),
    "love": (8.5, 6.5), "calm": (7.5, 2.8), "relaxed": (7.6, 2.5), "peaceful": (7.8, 2.6),
    "content": (7.7, 3.5), "sad": (2.5, 4.0), "unhappy": (3.0, 4.5), "depressed": (2.0, 3.8),
    "miserable": (1.8, 4.2), "angry": (3.5, 7.2), "mad": (3.2, 7.5), "furious": (2.8, 8.0),
    "irritated": (4.0, 6.5), "annoyed": (4.2, 6.0), "fearful": (3.0, 7.0), "afraid": (3.1, 6.8),
    "scared": (2.9, 7.3), "terrified": (2.5, 8.2), "hate": (2.7, 6.8), "bored": (3.2, 2.8),
    "shit": (3.0, 7.8),  # intense slang
}

def get_intensity(words, i):
    intensity = 1.0
    if i > 0 and words[i-1] in intensifiers:
        intensity *= intensifiers[words[i-1]]
    return intensity

def is_negated(words, i):
    for j in range(max(0, i-4), i):
        if words[j] in negation_words or words[j].endswith("n't"):
            return True
    return False

def has_negation(words):
    return any(w in negation_words or w.endswith("n't") for w in words)

def analyze_sentimento_cartesiano(text: str):
    text_lower = text.lower()
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s*', text)
    
    total_x = 0.0
    total_y = 0.0
    cumulative_points = [(0.0, 0.0)]
    
    for sentence in sentences:
        sentence_original = sentence
        sentence = sentence.lower().strip()
        if not sentence:
            continue
            
        words = re.findall(r"\b\w+(?:'\w+)?\b", sentence)
        has_neg_in_sentence = has_negation(words)
        
        for i, word in enumerate(words):
            cleaned = re.sub(r"[^a-z']", '', word)
            if len(cleaned) < 4:
                continue
                
            best_match, score = process.extractOne(cleaned, lexicon.keys(), scorer=fuzz.WRatio)
            if score > 82:
                intensity = get_intensity(words, i)
                v, a = lexicon[best_match]
                valence_base = v - 5.0
                arousal_base = a - 5.0
                
                if is_negated(words, i):
                    total_x += -valence_base * intensity
                    total_y += -arousal_base * intensity
                else:
                    total_x += valence_base * intensity
                    total_y += arousal_base * intensity
        
        # BOOST GRITO E EXCLAMAÇÃO (pra frases sem palavras emocionais mas cheias de energia)
        sentence_clean = sentence_original.strip()
        if sentence_clean:
            num_exclam = sentence_clean.count('!')
            is_upper = sentence_clean.isupper()
            arousal_boost = num_exclam * 2.5
            if is_upper:
                arousal_boost += 4.0
            total_y += arousal_boost
            
            if num_exclam > 0 and not has_neg_in_sentence:
                total_x += num_exclam * 1.2
        
        cumulative_points.append((total_x, total_y))
    
    return cumulative_points

if __name__ == "__main__":
    texto = "HEY ELON PUT THIS SHIT IN GROK!! ITS FROM MARS!!"
    
    points = analyze_sentimento_cartesiano(texto)
    final_x, final_y = points[-1]
    
    print(f"PONTO FINAL CARTESIANO: ({final_x:.2f}, {final_y:.2f})")
    print(f"Trajetória completa: {points}")
    
    # PLOT CHIQUE PRA MOSTRAR PRO PATRÃO
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    
    all_vals = [abs(v) for v in xs + ys]
    max_abs = max(all_vals) + 5 if all_vals else 15
    
    plt.figure(figsize=(12, 10))
    plt.plot(xs, ys, 'o-', color='cyan', linewidth=5, markersize=12, label='Trajetória do Sentimento')
    plt.scatter(0, 0, color='lime', s=300, zorder=5, label='Início (Neutro)')
    plt.scatter(xs[-1], ys[-1], color='red', s=400, marker='*', zorder=5, label='Final')
    
    plt.axhline(0, color='gray', linewidth=1.5, linestyle='--')
    plt.axvline(0, color='gray', linewidth=1.5, linestyle='--')
    plt.grid(True, alpha=0.4)
    
    quad_pos = 0.75 * max_abs
    quad_neg = -0.75 * max_abs
    
    plt.text(quad_pos, quad_pos, 'EXCITED\nHAPPY\nTHRILLED', ha='center', va='center', fontsize=16, fontweight='bold',
             bbox=dict(facecolor='lightgreen', alpha=0.7))
    plt.text(quad_neg, quad_pos, 'ANGRY\nFURIOUS\nFEARFUL', ha='center', va='center', fontsize=16, fontweight='bold',
             bbox=dict(facecolor='lightcoral', alpha=0.7))
    plt.text(quad_pos, quad_neg, 'CALM\nRELAXED\nPEACEFUL', ha='center', va='center', fontsize=16, fontweight='bold',
             bbox=dict(facecolor='lightblue', alpha=0.7))
    plt.text(quad_neg, quad_neg, 'SAD\nDEPRESSED\nBORED', ha='center', va='center', fontsize=16, fontweight='bold',
             bbox=dict(facecolor='lightgray', alpha=0.7))
    
    plt.xlim(-max_abs, max_abs)
    plt.ylim(-max_abs, max_abs)
    plt.xlabel('Valence ← Desprazer ——— Neutral ——— Prazer →', fontsize=14)
    plt.ylabel('Arousal ← Baixa ——— Neutral ——— Alta →', fontsize=14)
    plt.title(f'Trajetória do Sentimento no Plano Cartesiano\nFrase: "{texto}"', fontsize=18, pad=20)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('emotion_plot.png')
    plt.show()