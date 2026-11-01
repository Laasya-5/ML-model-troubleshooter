from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

PURPLE = colors.HexColor("#6d28d9")


def _table(rows, widths=None, header=True):
    t = Table(rows, colWidths=widths, hAlign="LEFT")
    st = [("FONTSIZE", (0, 0), (-1, -1), 9), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#ddd6f3")),
          ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), PURPLE), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
               ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    t.setStyle(TableStyle(st))
    return t


def build_pdf(path, data):
    ss = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=ss["Title"], textColor=PURPLE, fontSize=20, alignment=0)
    H2 = ParagraphStyle("H2", parent=ss["Heading2"], textColor=colors.HexColor("#3b2a6b"), spaceBefore=12)
    B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.5, leading=13)
    P = lambda s, st=B: Paragraph(str(s).replace("&", "&amp;").replace("<", "&lt;"), st)
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm, title="ML Model Troubleshooter report")
    s = [P("ML Model Troubleshooter", H1), P(f"{data['rtype']}  -  model: {data['model']['name']}"),
         P(f"Generated {data['generated']}  -  evaluation data: {data.get('eval_source') or 'none available'}"), Spacer(1, 6)]
    sec = data["sections"]
    if "Model Summary" in sec:
        m = data["model"]
        s.append(P("Model Summary", H2))
        s.append(_table([["Property", "Value"], ["Name", m["name"]], ["File", m["filename"]],
                         ["Framework (detected)", m["framework"]], ["Task", m["taskType"]],
                         ["Inputs", m["featureText"]], ["Health", data.get("health", "Unknown")]], [50 * mm, 120 * mm]))
        mt = data.get("metrics")
        if mt:
            s.append(Spacer(1, 6))
            if mt["task"] == "Regression":
                rows = [["R2", "MAE", "RMSE", "Rows"], [mt["r2"], mt["mae"], mt["rmse"], mt["n"]]]
            else:
                rows = [["Accuracy", "Precision", "Recall", "F1", "ROC AUC", "Rows"],
                        [mt["accuracy"], mt["precision"], mt["recall"], mt["f1"], mt.get("roc_auc", "-"), mt["n"]]]
            s.append(_table(rows))
    if "Validation & Check Results" in sec:
        s.append(P("Validation & Check Results", H2))
        s.append(_table([["Check", "Status", "Detail"]] + [[P(c["name"]), c["status"], P(c["detail"])] for c in data["checks"]],
                        [40 * mm, 20 * mm, 110 * mm]))
    if "Issues & Recommendations" in sec:
        s.append(P("Issues & Recommendations", H2))
        if not data["issues"]:
            s.append(P("No issues detected on the evaluation data."))
        for i in data["issues"]:
            s.append(P(f"[{i['severity']}] {i['title']} - {i['description']}"))
            s.append(P("Recommended: " + "; ".join(i["steps"])))
            s.append(Spacer(1, 3))
    if "Verification Results" in sec:
        s.append(P("Verification Results", H2))
        v = data.get("verification")
        if not v:
            s.append(P("This model has no parent version, so no before/after verification exists. Apply a fix from "
                       "Troubleshooting to create one."))
        else:
            s.append(P(f"Before: {v['before_name']}   After: {v['after_name']}   Verdict: {v['verdict']}"))
            rows = [["Metric", "Before", "After", "Change"]]
            for k, d in v["deltas"].items():
                rows.append([k, v["before"][k], v["after"][k], f"{d:+.4f}"])
            s.append(_table(rows))
    if "Monitoring & Logs" in sec:
        mo = data["monitor"]
        s.append(P("Monitoring & Logs", H2))
        s.append(_table([["Predictions logged", "Avg latency", "p95 latency", "Error rate", "Drift"],
                         [mo["total"], f"{mo['avg_latency']} ms" if mo["avg_latency"] is not None else "-",
                          f"{mo['p95_latency']} ms" if mo["p95_latency"] is not None else "-",
                          f"{mo['error_rate']*100:.1f}%", mo["drift"]["status"]]]))
        s.append(Spacer(1, 6))
        s.append(_table([["Time", "Event"]] + [[l["ts"], P(l["message"])] for l in data["logs"][:15]], [40 * mm, 130 * mm]))
    doc.build(s)
