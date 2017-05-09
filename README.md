# 2sigma
2Sigma Financial Competition on Kaggle

Used https://github.com/Giqles/kagglegym to create Docker container

Quicklaunch
```
git clone https://github.com/Giqles/kagglegym.git
docker build -t kagglegym .
docker run -it -v $(pwd):/wd -p 8888:8888 kagglegym jupyter notebook --port=8888 --ip=0.0.0.0
```
or without Jupyter
```
docker run -it -v $(pwd):/wd kagglegym
```
