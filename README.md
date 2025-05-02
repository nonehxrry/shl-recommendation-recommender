# SHL Assessment Recommendation Engine 🚀

This project is a smart, content-based recommendation system designed for SHL's assessment catalogue. It allows users (e.g., recruiters, hiring managers, or internal HR professionals) to input skill requirements or job roles and receive personalized recommendations from SHL's assessment database.

---

## 🔧 Features
- **Flask-based backend API** that uses TF-IDF and cosine similarity for content-based recommendations.
- **Streamlit frontend UI** for seamless user interaction.
- Accepts natural language inputs like "looking for a test to evaluate Java skills".
- Returns top 5 recommended assessments with detailed descriptions.

---

## 📁 Project Structure
```bash
.
├── app.py               # Flask backend API
├── ui.py                # Streamlit frontend UI
├── shl_catalogue_detailed.csv  # Dataset of SHL assessments
├── requirements.txt     # List of dependencies
└── README.md            # You're here :)
```

---

## 🛠️ Installation
1. **Clone this repository**
```bash
git clone https://github.com/nonehxrry/shl-recommendation-engine.git
cd shl-recommendation-engine
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Start Flask API** (in one terminal)
```bash
python app.py
```

4. **Start Streamlit UI** (in another terminal)
```bash
streamlit run ui.py
```

---

## 🔍 Sample Input
```
"Looking for a candidate strong in business analytics and data visualization"
```

## ✅ Sample Output
Top 5 recommendations:
1. SHL Data Analytics Test
2. Business Reasoning Assessment
3. Excel Proficiency Test
...

---

## 📊 Evaluation Strategy
- TF-IDF vectorization of combined assessment content
- Cosine similarity with user input
- Scores ranked to find top 5 most similar assessments

---

## 📈 Optimization Done
- Custom stopwords
- Expanded catalogue descriptions
- Combined multiple fields (skills, roles, descriptions) into a single vector

---

## 🧠 Future Enhancements
- Add keyword highlighting in results
- Include similarity score in UI
- Fine-tuned NLP with spaCy or SBERT for better context

---

## 📬 Contact
For questions or support, email: [harjtibhadauriya0610@gmail.com](mailto:harjtibhadauriya0610@gmail.com)

---

**Built with ❤️ for SHL Research Intern Assignment**
