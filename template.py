import numpy as np
import os
import json
import random

#-------------------------------------------------------------------
#--------------------FX attivazione---------------------------------
#-------------------------------------------------------------------
# ne ho messe alcune, softmax e tan sono funzioni logistiche, softmax serve per output binari , possono perdere il gradiente
# Poi ci sono le funzioni lineari :leaky relu è una versione migliorata di relu , cosi anche elu , funzioni lineari che possono far verificare esplosione del gradiente
# le funzioni lineari si usano per il deep learning , sono piu veloci nell'apprendimento e piu efficienti in termini di risultato 
def softmax(x):
    exp_values = np.exp(x - np.max(x))  # Gestisce il problema numerico per grandi valori di x
    return exp_values / np.sum(exp_values, axis=0, keepdims=True)
def leaky_relu(x, alpha=0.01):
    return np.maximum(alpha * x, x)
def tan(x):
   return np.tanh(x)
def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))
#-------------------------------------------------------------------
#-------------------------------------------------------------------

# importa pesi e bias e/o crea la memoria, è separata dal corpo della rete per gestirla in caso di addestramento
def crea_dizionario_liste(numero_di_liste):  # importa pesi e bias e/o crea la memoria, è separata dal corpo della rete per gestirla in caso di addestramento
        if os.path.exists("memoria.json"):
            with open("memoria.json", "r") as file:
                dizionario_liste = json.load(file)
        else:
            dizionario_liste = {}

            for i in range(1, numero_di_liste + 1):
                chiave_w = f"w{i}"
                chiave_b = f"b{i}"

                lista_w = [random.uniform(-1, 1) for _ in range(max(struttura))]
                lista_b = random.uniform(-1, 1)

                dizionario_liste[chiave_w] = lista_w
                dizionario_liste[chiave_b] = lista_b

            with open("memoria.json", "w") as file:
                json.dump(dizionario_liste, file, indent=2)

        return dizionario_liste

#uso questa semplice funzione per ficcare i valori tra lo zero e 1 in modo che moltiplicandoli in successione si ottiente sempre un valore inferiore a uno (si puo modificare a piacimento)
def normalizza_result(result):
    for i in range(len(result)):
        if result[i] > 1:
            result[i] /= 10
    return result

#tolglie i collegamenti neurali che sono esplosi, cerca le parole SOLO nel dizionario, non si puo usare in variabili e liste
def fix_overflow(dictionary):
    for key, value in dictionary.items():
        if isinstance(value, list):  # Se il valore è una lista
            # Sostituisci i valori nella lista che sono superiori a 1 miliardo o inferiori a -1 miliardo con 0.1
            dictionary[key] = [0.1 if v > 1e9 or v < -1e9 else v for v in value]
        else:  # Se il valore non è una lista
            try:
                float_value = float(value)  # Prova a convertire il valore in float
                # Se il valore è superiore a 1 miliardo o inferiore a -1 miliardo, sostituisci con 0.1
                if float_value > 1e9 or float_value < -1e9:
                    dictionary[key] = 0.1
                else:
                    dictionary[key] = float_value
            except (ValueError, TypeError):
                print(f"Il valore '{value}' della chiave '{key}' non può essere convertito in float.")
                dictionary[key] = 0.1  # Se non riuscito, sostituisci il valore con 0.1
    return dictionary

#corpo della RN
def Rete_neurale(inputs, struttura,FXa=leaky_relu, obbiettivo=0, LR=0.05, Keep_learning=True):
    x = 0#counters per gestire e capire a che strato siamo
    aa = 1
    for i in range(len(struttura)):#calcolo per ogni layer
        OUT_finale = []
        for u in range(1, struttura[x] + 1):#calcolo per ogni singolo neurone neurone
            weights = memoria_rete[f"w{aa}"] 
            bias = memoria_rete[f"b{aa}"]
            A = sum(g * l for g, l in zip(inputs, weights)) + bias #moltiplicazione pesi e bias
            A = FXa(A)  # FX attivazione

            if Keep_learning:#back propagation , si puo disattivare in caso la rete raggiunga un risultato ottimale per evitare l'esplosione del gradiente
                '''sqerr = (A - obbiettivo) ** 2''' # errore quadratico, se  interessa per monitorare l'apprendimento
                delta = 2 * (A - obbiettivo) * (1 - A ** 2)
                for b in range(len(inputs)):#aggiornamento singolo peso e bias
                    weights[b] = weights[b] - LR * delta * inputs[b]
                memoria_rete[f"b{aa}"] = bias - LR * delta
            aa = aa + 1#counters per gestire e capire a che strato siamo
            OUT_finale.append(A)
        '''normalizza_result(OUT_finale)''' #in caso serva, si potrebbe integrare qui una funzione per adattare i risultati (solo con fx lineari)
        inputs = OUT_finale# viene appoggiata la variabile all'input per riutilizzarla nel prossimo layer
        x = x + 1#counters per gestire e capire a che strato siamo
    return OUT_finale,memoria_rete

# Va creata la struttura , poi la memoria, poi l'input , e infine si esegue la rete
# la Struttura deve sempre essere una lista con numeri interi, il primo numero equivale al numero di neuroni del primo layer ecc.
struttura=[2,1]

#creare sempre prima la memoria (serve per la lettura se esistente)
memoria_rete=crea_dizionario_liste(sum(struttura))# in pratica crea un peso per ogni neurone, quindi gli devi dare il numero totale di neuroni

#l'input deve essere sempre una lista e il numero di inputs deve essere inferiore al numero di neuroni nel primo layer
input_per_la_rete=[2,3]

#siccome la funzione Rete_neurale da come output due variabili si puo creare contemporaneamente prima il risultato e poi la memoria
risultato_della_rete_neurale,la_Memoria_rete_da_salvare_o_riutilizzare=Rete_neurale(inputs=input_per_la_rete,struttura=struttura)

#piazzare il fix per overflow in caso si utilizzi le fx lineari
la_Memoria_rete_da_salvare_o_riutilizzare=fix_overflow(la_Memoria_rete_da_salvare_o_riutilizzare)

#salvare la memoria
with open("memoria.json", "w") as file:
    json.dump(la_Memoria_rete_da_salvare_o_riutilizzare, file, indent=2)

print('OUTPUT:',risultato_della_rete_neurale)
print('MEMORIA:',la_Memoria_rete_da_salvare_o_riutilizzare)

#per addestrarla basta creare un dataset e con un loop for eseguire la rete in modo che prenda gli input in sequenza