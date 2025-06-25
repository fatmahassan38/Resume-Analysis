from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import PyPDF2
import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(720, 520)
        Form.setStyleSheet("""
            QWidget {
                background-color: #f0f4f8;
                font-family: Arial;
                font-size: 12pt;
            }
            QPushButton {
                padding: 5px;
                font-weight: bold;
                text-align: center;
                cursor: pointer;
            }
        """)

        self.titleLabel = QtWidgets.QLabel(Form)
        self.titleLabel.setGeometry(QtCore.QRect(20, 5, 680, 30))
        self.titleLabel.setObjectName("titleLabel")
        self.titleLabel.setText("Resume Analyzer")
        self.titleLabel.setStyleSheet("font-size: 18pt; font-weight: bold; color: #333;")
        self.titleLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.jobInput = QtWidgets.QLineEdit(Form)
        self.jobInput.setGeometry(QtCore.QRect(20, 45, 680, 30))
        self.jobInput.setObjectName("jobInput")
        self.jobInput.setPlaceholderText("Enter job description here...")
        self.jobInput.setStyleSheet("padding: 5px; border: 1px solid #a0a0a0; border-radius: 6px;")

        self.loadPDFBtn = QtWidgets.QPushButton(Form)
        self.loadPDFBtn.setGeometry(QtCore.QRect(20, 85, 150, 35))
        self.loadPDFBtn.setObjectName("loadPDFBtn")
        self.loadPDFBtn.setText("Load PDF")
        self.loadPDFBtn.setStyleSheet("""
            QPushButton {
                background-color: #0078d7;
                color: white;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #005bb5;
            }
        """)
        self.loadPDFBtn.clicked.connect(self.load_pdf)

        self.fileLabel = QtWidgets.QLabel(Form)
        self.fileLabel.setGeometry(QtCore.QRect(180, 85, 520, 30))
        self.fileLabel.setObjectName("fileLabel")
        self.fileLabel.setStyleSheet("padding-left: 5px;")

        self.similarityBtn = QtWidgets.QPushButton(Form)
        self.similarityBtn.setGeometry(QtCore.QRect(20, 130, 150, 35))
        self.similarityBtn.setObjectName("similarityBtn")
        self.similarityBtn.setText("Check Similarity")
        self.similarityBtn.setStyleSheet("""
            QPushButton {
                background-color: #0078d7;
                color: white;
                border-radius: 6px;
                font-size: 10pt;
            }
            QPushButton:hover {
                background-color: #005bb5;
            }
        """)
        self.similarityBtn.clicked.connect(self.check_similarity)

        self.similarityOutput = QtWidgets.QTextEdit(Form)
        self.similarityOutput.setGeometry(QtCore.QRect(20, 170, 680, 100))
        self.similarityOutput.setObjectName("similarityOutput")
        self.similarityOutput.setStyleSheet("background-color: #ffffff; border: 1px solid #a0a0a0; border-radius: 6px; padding: 5px;")

        self.entityBtn = QtWidgets.QPushButton(Form)
        self.entityBtn.setGeometry(QtCore.QRect(20, 280, 150, 35))
        self.entityBtn.setObjectName("entityBtn")
        self.entityBtn.setText("Get Summary")
        self.entityBtn.setStyleSheet("""
            QPushButton {
                background-color: #0078d7;
                color: white;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #005bb5;
            }
        """)
        self.entityBtn.clicked.connect(self.summarize_text)

        self.entityOutput = QtWidgets.QTextEdit(Form)
        self.entityOutput.setGeometry(QtCore.QRect(20, 320, 680, 170))
        self.entityOutput.setObjectName("entityOutput")
        self.entityOutput.setStyleSheet("background-color: #ffffff; border: 1px solid #a0a0a0; border-radius: 6px; padding: 5px;")

    def load_pdf(self):
        options = QtWidgets.QFileDialog.Options()
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Open PDF File", "", "PDF Files (*.pdf)", options=options)
        if file_path:
            self.fileLabel.setText(f"Loaded: {os.path.basename(file_path)}")
            self.loaded_text = self.extract_text_from_pdf(file_path)
            if "Error" in self.loaded_text:
                self.similarityOutput.setText(self.loaded_text)

    def extract_text_from_pdf(self, pdf_path):
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + " "
            return text.strip()
        except Exception as e:
            return f"Error reading PDF: {str(e)}"

    def preprocess(self, text):
        stop_words = set(stopwords.words('english'))
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s|]', '', text)
        words = word_tokenize(text)
        words = [w for w in words if w not in stop_words]
        return ' '.join(words)

    def check_similarity(self):
        try:
            job_desc = self.jobInput.text()
            if hasattr(self, 'loaded_text') and job_desc:
                resume_clean = self.preprocess(self.loaded_text)
                job_desc_clean = self.preprocess(job_desc)
                cntv = CountVectorizer()
                count_matrix = cntv.fit_transform([resume_clean, job_desc_clean])
                percentage = round((cosine_similarity(count_matrix)[0][1] * 100), 2)
                self.similarityOutput.setText(f"Similarity with Job Description: {percentage}%")
            else:
                self.similarityOutput.setText("Please load a resume and enter job description first.")
        except Exception as e:
            self.similarityOutput.setText(f"Error during similarity check: {str(e)}")

    def summarize_text(self):
        if hasattr(self, 'loaded_text'):
            summary = self.loaded_text[:700] + "..." if len(self.loaded_text) > 700 else self.loaded_text
            self.entityOutput.setText(summary)
        else:
            self.entityOutput.setText("Please load a resume first.")

if __name__ == "__main__":
    nltk.download('punkt')
    nltk.download('stopwords')

    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())