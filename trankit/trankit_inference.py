import trankit
from trankit import Pipeline

# trankit.verify_customized_pipeline(
#     category='customized-ner', # pipeline category
#     save_dir='./output_dir' # directory used for saving models in previous steps
# )

p = Pipeline(lang='customized-ner', cache_dir='./output_dir')
#res = p.ner('''Vlani súd v Bratislave vymeral Marošovi Deákovi 25 rokov väzenia za rôzne zločiny''', is_sent=True)
res = p.ner('''Vláda už minula miliardovú rezervu v rozpočte, ktorá bola určená na krytie výdavkov súvisiacich s pandémiou.
            Minister financií a predseda OĽaNO Igor Matovič preto predložil návrh na ďalšie zvýšenie výdavkov rozpočtu.''')

print(res)