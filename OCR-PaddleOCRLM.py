from paddleocr import DocVLM

model = DocVLM(model_name="PP-DocBee-2B")
results = model.predict(
    input={
        "pdf": "D:\OneFile\WorkOnly\AllCode\GLSWork\Typhoon_OCR\shared\input.pdf",
    },
    batch_size=1,
)
for res in results:
    res.print()
    res.save_to_json(f"./output/res.json")
