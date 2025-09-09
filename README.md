# 🎓 Placement Chatbot

A simple **RAG-based PDF Q&A chatbot** that answers placement-related questions directly from official placement PDFs (e.g. IGDTUW placement reports). Instead of scrolling through pages, just ask questions like:

* *“What was the highest package in 2023?”*
* *“Average CTC for MCA in 2022?”*

…and get instant answers 🚀

---

## ✨ Features

* 📂 Reads & processes **placement PDFs**
* 🧠 Uses **Google Gemini + FAISS** for retrieval & QA
* 💬 **Streamlit chatbot UI**
* 📊 Extracts metrics like highest CTC, average CTC, recruiters, students placed
* 🎯 Auto-detects **year & department** from your query

---

## 🗂 Project Structure

```
.
├── app.py            # Core logic: parsing, embeddings, QA
├── chatbot_ui.py     # Streamlit chatbot interface
├── Placement Data/   # Put your placement PDFs here
├── .env              # Store GOOGLE_API_KEY here
├── requirements.txt  # Dependencies
```

---

## ⚡ Quick Start

1. Clone the repo:

   ```bash
   git clone https://github.com/Mansi-Tanwar/RAG_based_PDF_QA.git
   cd RAG_based_PDF_QA
   ```

2. Create and activate virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate      # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Add your Google API key in `.env`:

   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

5. Place PDFs inside **Placement Data/** folder.

6. Run the chatbot:

   ```bash
   streamlit run chatbot_ui.py
   ```

7. Open [http://localhost:8501](http://localhost:8501) 🎉

---

## 🐞 Troubleshooting

* **NLTK error** → Ensure `nltk.download('punkt')` is done.
* **FAISS issue** → Install `faiss-cpu` (`pip install faiss-cpu`).
* **Gemini error** → Check your API key & billing settings.

---

## 🚀 Future Improvements

* Add **charts/visualizations** for recruiters & salary trends
*Add OCR support so that even scanned placement PDFs (image-based) can be processed and understood
* Cache embeddings for faster startup
* Show **sources (PDF + page)** in responses
* Add a refresh option for reloading PDFs without restart

---

## 📌 Notes

* Works best with **clean PDFs** (not scanned).
* File names should include the **year** (e.g. `IGDTUW_Placement_2023.pdf`).

---

Would you like me to also make a **shorter portfolio-style version** (1–2 sections only, good for recruiters to skim), or keep this as the main detailed README?
