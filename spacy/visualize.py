import spacy
from spacy import displacy

j = {
       "text": "Vzniká severne od mesta Dunboyne pri osade Killester.",
       "ents": [
           {"start": 23, "end": 32, "label": "LOC"},
       ],
       "title": None
     }

# in Jupyter notebook
#displacy.render(doc, style="ent", jupyter=True, options={'colors': {'PER': 'yellow', 'MISC': '#f89ff8'}})

# save as HTML
html = displacy.render(j, manual=True, style="ent", page=True, options={'colors': {'PER': 'yellow', 'MISC': '#f89ff8'}})
with open('visualized.html', 'w') as f:
    f.write(html)