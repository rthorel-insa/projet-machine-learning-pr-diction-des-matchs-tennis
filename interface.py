# interface.py
import tkinter as tk
from tkinter import ttk, messagebox
import demarrage # On importe ton fichier métier !

class AppTennis:
    def __init__(self, root):
        self.root = root
        self.root.title("🎾 IA Pronostics ATP")
        self.root.geometry("400x450")
        self.root.config(padx=20, pady=20)

        # --- Variables pour stocker le modèle ---
        self.model = None
        self.columns = None
        self.df_global = None
        self.forme_5 = None
        self.h2h = None

        # --- Écran de chargement ---
        self.label_statut = tk.Label(root, text="Initialisation de l'IA...\nCela peut prendre quelques secondes.", font=("Arial", 12), fg="blue")
        self.label_statut.pack(pady=50)
        
        # On force la fenêtre à s'afficher avant de lancer le calcul lourd
        self.root.update() 
        self.charger_modele()

    def charger_modele(self):
        # On appelle le service métier !
        self.model, self.columns, self.df_global, self.forme_5, self.h2h, erreur = moteur_tennis.entrainer_modele()
        
        if erreur:
            messagebox.showerror("Erreur", erreur)
            self.root.destroy()
        else:
            self.label_statut.destroy() # On cache le message de chargement
            self.afficher_interface()   # On affiche les boutons

    def afficher_interface(self):
        # Titre
        tk.Label(self.root, text="Paramètres du match", font=("Arial", 14, "bold")).pack(pady=10)

        # Joueur 1
        tk.Label(self.root, text="Joueur 1 :").pack()
        self.entree_joueur_a = tk.Entry(self.root, width=30)
        self.entree_joueur_a.insert(0, "Etcheverry")
        self.entree_joueur_a.pack(pady=5)

        # Joueur 2
        tk.Label(self.root, text="Joueur 2 :").pack()
        self.entree_joueur_b = tk.Entry(self.root, width=30)
        self.entree_joueur_b.insert(0, "Bergs")
        self.entree_joueur_b.pack(pady=5)

        # Surface
        tk.Label(self.root, text="Surface :").pack()
        self.combo_surface = ttk.Combobox(self.root, values=["Hard", "Clay", "Grass"], state="readonly")
        self.combo_surface.current(0) # Sélectionne "Hard" par défaut
        self.combo_surface.pack(pady=5)

        # Bouton
        btn_predire = tk.Button(self.root, text="Lancer la prédiction", bg="#28a745", fg="white", font=("Arial", 12, "bold"), command=self.clic_predire)
        btn_predire.pack(pady=20)

        # Résultat
        self.label_resultat = tk.Label(self.root, text="", font=("Arial", 12), justify="center")
        self.label_resultat.pack(pady=10)

    def clic_predire(self):
        nom_a = self.entree_joueur_a.get().strip()
        nom_b = self.entree_joueur_b.get().strip()
        surface = self.combo_surface.get()

        if not nom_a or not nom_b:
            messagebox.showwarning("Attention", "Veuillez entrer le nom des deux joueurs.")
            return

        # On appelle la fonction métier pour faire la prédiction !
        resultats, erreur = moteur_tennis.predire_match(
            nom_a, nom_b, surface, 
            self.model, self.columns, self.df_global, self.forme_5, self.h2h
        )

        if erreur:
            messagebox.showerror("Erreur", erreur)
        else:
            # On met à jour l'interface avec la réponse
            texte = f"{nom_a} : {resultats['prob_a']:.1f}%\n\nContre\n\n{nom_b} : {resultats['prob_b']:.1f}%"
            self.label_resultat.config(text=texte, fg="black")

# Lancement de l'application
if __name__ == "__main__":
    fenetre = tk.Tk()
    app = AppTennis(fenetre)
    fenetre.mainloop()