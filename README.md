# ipynb파일을 md파일로 변환

source: https://www.fast.ai/2020/01/20/nb2md/

```sh
pip install nbdev
```

노트북 파일이 들어있는 폴더에서 아래 명령어 실행 (여기서는 name.ipynb라는 이름을 지니고 있다고 가정한다)

>  주의! nbdev_nb2md 실행시 폴더안에 settings.ini파일이 있어야 함.

```sh
nbdev_nb2md name.ipynb
```

name.md파일과 노트북에 이미지가 있다면 /images/폴더에 저장됨

이미지가 있을 경우 upd_md.py를 실행해서 이미지파일 경로 앞에 `/images/`접두사를 붙여줌.
