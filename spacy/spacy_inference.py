import spacy

nlp1 = spacy.load("./output/model-best") #load the best model
doc = nlp1('''Vláda už minula miliardovú rezervu v rozpočte, ktorá bola určená na krytie výdavkov súvisiacich s pandémiou. 
              Minister financií a predseda OĽaNO Igor Matovič preto predložil návrh na ďalšie zvýšenie výdavkov rozpočtu.''')

doc.to_disk('spacy_inference_result.txt')
#spacy.displacy.render(doc, style="ent", jupyter=True) # display in Jupyter