import newspaper
import pandas as pd

REAL_SOURCES = [
  "https://www.nation.africa",
  "https://www.standardmedia.co.ke",
  "https://www.the-star.co.ke",
  "https://www.kbc.co.ke/"
]

FAKE_SOURCES = [
  "https://www.kenyannewspropaganda.com",
  "https://africacheck.org/"
  
]

def collect_articles(sources, label):
  articles = []
  for url in sources:
      paper = newspaper.build(url, memoize_articles=False)
      for article in paper.articles[:50]:  # Limit for speed/testing
          try:
              article.download()
              article.parse()
              if len(article.text) > 200:  # Only keep substantial articles
                  articles.append({'text': article.title + ". " + article.text, 'label': label})
          except Exception as e:
              continue
  return articles

real_articles = collect_articles(REAL_SOURCES, 1)
fake_articles = collect_articles(FAKE_SOURCES, 0)
all_articles = real_articles + fake_articles

df = pd.DataFrame(all_articles)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv('kenyan_news.csv', index=False)
print("Collected news saved to kenyan_news.csv")