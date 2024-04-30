# Projet 7 - Parcours Data Scientist (OpenClassrooms)

Ce repo contient des livrables du projet 7 de la formation *Data Scientist* proposée par *OpenClassrooms*. Vous y trouverez notamment :
- **Modelisation.ipynb** : le notebook pour le pré-traitement et la modélisation des données.
- **model_deployment.py** : le code pour déployer le modèle sous forme d'API.
- **dashboard.py** : le code du dashboard utilisant la librairie `streamlit`.


## Déploiement en local

- Installez les librairies nécessaires avec la commande ``pip install -r requirements.txt``
- Lancez le fichier ``model_deployment.py`` (en tapant `python model_deployment.py` dans le terminal par exemple)
- Remplacez la valeur de ``model_uri`` dans le fichier `dashboard.py` par l'adresse renvoyée par la commande 
précédente (cela devrait être ``http://127.0.0.1:80``)
- Lancez la commande ``streamlit run dashboard.py`` depuis le terminal


## Déploiement sur le Cloud AWS

Vous pouvez déployer le modèle sur le cloud AWS en suivant les étapes décrites ci-dessous :

- Lancez une instance EC2 sur AWS. Modifiez le groupe de sécurité de sorte à autoriser le trafic HTTP sur le port
80 (ou tout autre port que vous choisirez pour déployer le modèle) pour que votre modèle soit accessible par tous depuis internet.
- Transférez les fichiers `Dockerfile`, `model_deployment.py`, `requirements.txt`, `pipeline_projet7.joblib` et
`explainer.dill`. Pour cela, ouvrez un terminal, rendez-vous dans le dossier du projet et lancez la commande suivante :
```
scp -i </path/to/my-key-pair.pem> Dockerfile model_deployment.py requirements.txt pipeline_projet7.joblib explainer.dill ec2-user@<public-dns-name>:</path/to/folder>
```
où `</path/my-key-pair.pem>` est le chemin menant vers votre clé d'accès, `<public-dns-name>` est
l'adresse DNS publique de votre instance EC2 et `</path/to/folder>` le chemin vers le dossier dans lequel vous
souhaitez enregistrer les fichiers sur votre instance.
- Installez ``Docker`` sur votre instance EC2. Si vous avez lancé une instance **Amazon Linux**, connectez-vous à votre instance et taper les commandes 
suivantes :
```
sudo amazon-linux-extras install docker
sudo yum install docker
sudo service docker start
sudo usermod -a -G docker ec2-user
```
- Toujours dans votre instance, lancez la commande suivante pour construire une image Docker :
```
docker build -t <image-name> .
```
en remplaçant `<image-name>` par un nom de votre choix.
- Lancez un container docker à partir de l'image créée avec la commande 
```
docker run -p 80:80 <image-name>
```

Vous pouvez maintenant accéder à votre modèle et lancer le dashboard en remplaçant la valeur de ``model_uri`` dans
le script de ``dashboard.py`` par l'adresse publique de votre instance.

Les données se trouvent dans un bucket AWS publique que j'ai créé. Si vous souhaitez modifier les données et les mettre sur
AWS S3, vous pouvez suivre les instructions sur ce [lien](https://www.simplified.guide/aws/s3/create-public-bucket)
pour créer un bucket publique. Vous pourrez ensuite charger vos données dans la fonction ``load_X_y`` du fichier
``dashboard.py`` en remplaçant les adresses par celles de vos données.


## Améliorations à envisager

- Pour la modélisation :
  - Essayer le **target encoding** plutôt que le one hot**.
  - Réduire le nombre de variables sélectionnées en réduisant le seuil de corrélation et la variance minimale.
  - Essayer d'autres fonctions pour l'**undersampling** (`EditedNearestNeighbours`, `NearMiss`, `TomekLinks`, ...).

- Pour le dashboard :
  - retracer le graphe quand l'échelle des abscisses sélectionnée change et pas seulement changer l'échelle de l'axe.


## Ressources

- https://towardsdatascience.com/simple-way-to-deploy-machine-learning-models-to-cloud-fd58b771fdcf
- Données : https://www.kaggle.com/c/home-credit-default-risk/data
