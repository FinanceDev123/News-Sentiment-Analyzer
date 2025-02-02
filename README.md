# 📊 News Sentiment Analyzer 📰📉  
A Python project using VADER to analyze sentiment in financial news and compare it to stock price movements.

This project is a **News Sentiment Analyzer** that processes financial news articles, applies **sentiment analysis using VADER**, and **compares sentiment trends to stock price movements**. The goal is to **help traders, analysts, and researchers identify correlations between news sentiment and market behavior**.  

---

## 🚀 **Project Overview**  

Financial markets are heavily influenced by **news, investor sentiment, and public perception**. This Python-based tool aims to:  
✅ **Fetch financial news articles** using the NewsAPI.  
✅ **Analyze sentiment using the VADER model**, scoring news as **positive, negative, or neutral**.  
✅ **Retrieve historical stock price data** using `yfinance`.  
✅ **Visualize the relationship between sentiment and stock prices** using Matplotlib.  
✅ Provide a structured **list of analyzed articles**, including **title, source, and date** for further investigation.  

---

## 🏗 **How It Works**  

### **1️⃣ Fetching News Articles 📰**  
- The script uses **NewsAPI** to fetch the latest financial news articles related to a given stock or topic.  
- The news headlines and descriptions are extracted for sentiment analysis.  

### **2️⃣ Sentiment Analysis using VADER 🤖**  
The **VADER (Valence Aware Dictionary and sEntiment Reasoner)** model is used to determine the sentiment of each news article.  

🔹 **How VADER Works:**  
- It **assigns sentiment scores** to words using a **predefined lexicon**.  
- It considers **punctuation, capitalization, and negation words** (e.g., *"not bad"* is different from *"bad"*).  
- It **calculates a compound sentiment score** ranging from **-1 (very negative) to +1 (very positive)**.  

✅ **Why Use VADER?**  
- It’s **fast, rule-based, and doesn't require training data**.  
- Works well on **short-form** text like **news headlines and tweets**.  
- Handles **intensifiers (e.g., "very good") and negations ("not bad")** better than simple keyword-based approaches.  

### **3️⃣ Retrieving Stock Data 📈**  
- The **Yahoo Finance API (`yfinance`)** is used to pull **historical stock prices** for the analyzed period.  
- Closing prices are **plotted against sentiment scores** to detect potential correlations.  

### **4️⃣ Data Visualization 📊**  
- **Stock Prices Over Time:** Displays the stock's closing prices over the selected period.  
- **News Sentiment Analysis:** A bar chart visualizing the sentiment scores of top articles.  
- **Sentiment vs Stock Price Correlation:** A dual-axis chart showing daily average sentiment alongside stock price movements.  

### **5️⃣ Listing Articles for Review 📜**  
- At the end of execution, the script **prints all fetched articles** in a structured format, showing:  
  - **Title**  
  - **Source**  
  - **Publication Date (YYYY/MM/DD)**  

---

## 📌 **Installation & Usage**  

### **🔹 Prerequisites**  
Ensure you have **Python 3.x** installed along with the following dependencies:  

```bash
pip install requests yfinance nltk matplotlib python-dateutil


## Thank you for your attention.🙏
