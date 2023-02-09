# Projet 7 - Parcours Data Scientist (OpenClassrooms)

Ce repo contient des livrables du projet 7 de la formation Data Scientist chez OpenClassrooms.
On peut y trouver les codes pour le déploiement du modèle et du dashboard. On y trouve notamment :
- Modelisation.ipynb : le notebook pour le pré-traitement et la modélisation des données.
- model_deployment.py : le code pour déployer le modèle sous forme d'API.
- dashboard.py : le code du dashboard déployé via `streamlit`.


## Déploiement en local

- Installez les librairies de ``requirements.txt`` (avec ``pip install -r requirements.txt`` par 
exemple)
- Lancer le fichier ``model_deployment.py`` (avec `python model_deployment.py` depuis le terminal par exemple)
- Remplacez la valeur de ``model_uri`` dans le fichier `dashboard.py` par l'adresse renvoyée par la commande 
précédente (cela devrait être ``http://127.0.0.1:80``)
- Lancez la commande ``streamlit run dashboard.py`` depuis le terminal


## Déploiement sur le Cloud

Vous pouvez accéder à l'application en cliquant sur le lien ci-dessous :

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://gabriel1628-projet-7-dashboard-umy46h.streamlit.app)

Le dashboard peut aussi être lancé depuis le terminal avec la commande `streamlit run dashboard.py`
(streamlit doit être installé). Le dashboard peut mettre plusieurs minutes à se lancer la première fois.

Si cela ne fonctionne plus, c'est que j'ai sûrement arrêté l'instance AWS EC2 qui héberge le modèle.
Dans ce cas, vous pouvez déployer le modèle en local et utiliser votre adresse locale, ou bien vous pouvez
déployer vous-même le modèle, pour cela :
- lancez une instance EC2 sur AWS. Modifiez le groupe de sécurité de sorte à autoriser le trafic HTTP sur le port
80 (ou tout autre port que vous choisirez pour déployer le modèle) pour que votre modèle soit accessible par tous
- transférez les fichiers `Dockerfile`, `model_deployment.py`, `requirements.txt`, `pipeline_projet7.joblib` et
`explainer.dill`. Pour cela, utilisez la commande

`scp -i /path/my-key-pair.pem file-to-copy [file-to-copy...] ec2-user@public-dns-name:/path/to/folder`

où `/path/my-key-pair.pem` est le chemin vers votre clé enregistré sur votre ordinateur, `public-dns-name` est
l'adresse DNS publique de votre instance EC2 et `/path/to/folder` le chemin vers le dossier dans lequel vous
souhaitez enregistrer les fichiers sur votre instance.
- installez ``Docker``. Si vous avez lancé une instance __Amazon Linux__, il vous suffit de taper les commandes 
suivantes :
```
sudo amazon-linux-extras install docker
sudo yum install docker
sudo service docker start
sudo usermod -a -G docker ec2-user
```
- toujours dans votre instance, lancez la commande suivante pour construire une image Docker :
`docker build -t <image-name> .`
- lancez un container docker à partir de l'image créée avec la commande 
`docker run -p 80:80 <image-name> .`

Vous pouvez maintenant accéder à votre modèle et lancer le dashboard en remplaçant la valeur de ``model_uri`` dans
le script de ``dashboard.py`` par l'adresse publique de votre instance.

Les données se trouvent dans un bucket AWS que j'ai créé. Si vous souhaitez modifier les données et les mettre sur
AWS S3, vous pouvez suivre les instructions sur ce [lien](https://www.simplified.guide/aws/s3/create-public-bucket)
pour créer un bucket publique. Vous pourrez ensuite charger vos données dans la fonction ``load_X_y`` du fichier
``dashboard.py`` en remplaçant les adresses par celles de vos données.


## Améliorations à faire

- Pour la modélisation :
  - Essayer le __target encoding__ plutôt que le __one hot__
  - Réduire le nombre de variables sélectionnées en réduisant le seuil de corrélation et la variance minimale
  - Essayer d'autres fonctions pour l'__undersampling__ (`EditedNearestNeighbours`, `NearMiss`, `TomekLinks`, ...)

- Pour le dashboard :
  - retracer le graphe quand l'échelle des abscisses sélectionnée change et pas seulement changer l'échelle de l'axe


## Ressources

- https://towardsdatascience.com/simple-way-to-deploy-machine-learning-models-to-cloud-fd58b771fdcf
