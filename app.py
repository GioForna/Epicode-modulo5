import gradio as gr

from src.chefbot import ChefBot
from src.knowledge import format_card

bot = ChefBot()


def classify(image):
    if image is None:
        return "Carica una foto di un piatto.", {}
    result = bot.recognize(image, top_k=5)
    card = format_card(result["info"], result["confidence"])
    if result["confidence"] < 0.4:
        card = "_Non sono molto sicuro, controlla anche le altre ipotesi._\n\n" + card
    return card, {bot.kb[label]["nome"]: p for label, p in result["top_k"]}


def mood(query):
    if not query.strip():
        return "Scrivi cosa ti va di mangiare."
    results = bot.mood_search(query, top_k=3)
    best = results[0]
    others = "\n".join(f"- {r['nome']} (similarità {r['score']:.2f})" for r in results[1:])
    return (
        f"**Ti consiglio: {best['nome']}** (similarità {best['score']:.2f})\n\n"
        + format_card(best["info"])
        + f"\n\n### Altre proposte\n{others}"
    )


with gr.Blocks(title="ChefBot AI") as demo:
    gr.Markdown("# ChefBot AI - Dallo scatto alla ricetta")
    with gr.Tab("Riconosci il piatto"):
        with gr.Row():
            img = gr.Image(type="pil", label="Foto del piatto")
            with gr.Column():
                probs = gr.Label(num_top_classes=5, label="Predizioni")
                card = gr.Markdown()
        gr.Button("Analizza").click(classify, inputs=img, outputs=[card, probs])
    with gr.Tab("Ricerca per umore"):
        query = gr.Textbox(label="Come ti senti? Cosa ti va?", placeholder="voglio qualcosa di fresco e leggero")
        answer = gr.Markdown()
        gr.Button("Cerca").click(mood, inputs=query, outputs=answer)
        query.submit(mood, inputs=query, outputs=answer)
        gr.Examples(
            ["voglio qualcosa di fresco e leggero", "dolce con miele", "ho freddo, mi serve qualcosa di caldo", "piatto piccante"],
            inputs=query,
        )

if __name__ == "__main__":
    demo.launch()
