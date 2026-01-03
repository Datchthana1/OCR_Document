from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_doc_orientation_classify=True,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    det_db_box_thresh=0.6,
    det_db_unclip_ratio=1.5,
    rec_batch_num=6,
    lang="th",
)
result = ocr.predict(
    rf"C:\Users\me095\Downloads\การพยากรณ์ความเข้มข้นของ-PM2.5-รายชั่วโมงและรายวัน-LSTM-VAR.pdf"
)
for res in result:
    res.print()
    res.save_to_img("D:\OneFile\WorkOnly\AllCode\GLSWork\Typhoon_OCR\output")
    res.save_to_json("D:\OneFile\WorkOnly\AllCode\GLSWork\Typhoon_OCR\output")
