'''

Einstieg in Python Klassen (Class)

'''


from random import randrange

import numpy as np
import pandas as pd




class Eisladen:

    def __init__(self, 
            ort='',
            name='', 
            sorten = ('vanille', 'schoko', 'erbeer')
        ):
        self.ort = ort
        self.name = name
        self.sorten = sorten
            

    def __repr__(self):
        txt = f'{self.__class__.__qualname__}(ort={self.ort}, name={self.name}, sorten={", ".join(self.sorten)})'
        return txt



eisladen_A = Eisladen(ort='Hengasch',
                      name='Drei Kugeln Haus', 
                      sorten = ('vanille', 'schoko', 'banane')
                      )

eisladen_B = Eisladen(ort='Standort',
                      name='Sortenspirale', 
                      )



class EisladenSammlung:
        
        
    def __init__(self, eislaeden):
        self.eislaeden = eislaeden
        
        self.frame = self.to_frame()
    
    def to_frame(self):
        list_dicts = [{'Variable': i, 'name': i.name, 'ort': i.ort, 'sorten': i.sorten} for i in self.eislaeden]
        return pd.DataFrame(list_dicts)
        
    
sammlung = EisladenSammlung((eisladen_A, eisladen_B))
print(sammlung.frame)



def beispiel_daten_erzeugen():
    
    
    monaten_im_jahr = np.arange(1,13)
    
    roh_daten = {
            'Monat' : [i for i in monaten_im_jahr],
            'chance_auf_schmelzen' : [beispeil_zahl_pro_monat(i) for i in monaten_im_jahr]
             }
    
    daten = convert_to_dataframe(roh_daten)
    
    return daten

       
def beispeil_zahl_pro_monat(monat):
    return np.sin((monat/12)*np.pi*randrange(100,130)/100 )*100

def convert_to_dataframe(daten):
    df = pd.DataFrame(daten)
    return df

def plot_beispiel_daten(df):
    return df.plot(x='Monat', y='chance_auf_schmelzen')

def erzeug_beispiel():
    
    daten = beispiel_daten_erzeugen()
    plot_beispiel_daten(daten)
    
if __name__ == '__main__':
    erzeug_beispiel()

