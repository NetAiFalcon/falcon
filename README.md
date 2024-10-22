## Infra.

![poster](./readme_assets/infra_now.png)

<details>
<summary>Previous</summary>
<div markdown="1">

![poster](./readme_assets/infra_previous.png)

초기 진행에 관한 자료 링크

[falcon에 관한 초기 아이디어](https://github.com/NetAiFalcon/falcon/tree/nats/initial_meterial)

</div>
</details>

## Why NATS?

선택 이유 : 스테이션 30대 가량이 한번에 들어오면 몰리는 감이 있어서 분리하고 싶은데
그렇다고 그룹을 바로 나누기에는 후에 데이터를 집중시킬 때 Top-Down이 잘 생각이 나지 않아서
NATS의 Subject Mapping and Partitioning 이 강점을 보임

1. Top-Down에 유리한 subject 세팅이 가능
2. 그 중에서도 특정 부분만을 받기에도 유용 --> 유연성이 다른 시스템에 비해 유용하다고 여김
3. 개인적으로 제일 많이 써본 메세지큐이라 익숙하고, MobileX의 하위 표준이기 때문

![poster](./readme_assets/falcon_subject.png)

위 그림과 같이 유연하게 원하는 부분만 구독하는 점을 강점으로 NATS를 채택함.  
(+ 설정이 편하고 가볍다)

## pub

python3 main.py --tag_id {int}

## Sub

NATS를 경유해서 값을 받고, kafka를 이용해 토픽을 전달
(falcon은 NATS 중심의 메세지큐를 사용하지만 MobileX 및 Digital Twin은 kafka를 사용하기 때문.)

현재는 미들웨어 처럼 작동하지만, 후에는 그룹장이 sub일을 맡고, 정제된 데이터를 한 번더 정제하는 최종 타워가 있을 것임. (previous work falcon pdf/ppt 참고)

## Dockerfile

한 번에 도커파일을 빌드할 경우 1000초 이상의 시간이 소요되기 때문에

falcon-base

falcon-pub

로 나누어 빌드함.

결과적으로 기존 1000s 이상 걸리는 빌드를 600s -> 400s -> < 3s 으로 단축함.

![poster](./readme_assets/falcon-dockerfile.png)

### Dockerfile build

```bash
sudo docker build -t falcon-base-torch-12.1-pub-base -f dockerfile_falcon-base .
```

```bash
sudo docker build -t falcon-pub -f dockerfile_falcon-pub .
```

### Docker run

```bash
sudo docker run -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all --net=host --env="DISPLAY" --device=/dev/snd:/dev/snd --device=/dev/video0:/dev/video0 --device=/dev/video1:/dev/video1 --device=/dev/media0:/dev/media0 -i -t -v /etc/localtime:/etc/localtime:ro -v /usr/lib:/usr/lib --gpus=all --replace --name=falcon-pub-tset minjuncho/falcon-pub python3 main.py --tag_id {uwb_tag_id}
```

**For demo**
```bash
sudo docker run -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all --net=host --env="DISPLAY" -e NATS_SERVER='nats server ip' -e NATS_PORT='4222'  --device=/dev/snd:/dev/snd --device=/dev/video0:/dev/video0 --device=/dev/video1:/dev/video1 --device=/dev/media0:/dev/media0 -i -t -v /etc/localtime:/etc/localtime:ro -v /usr/lib:/usr/lib --gpus=all --name=falcon-pub-tset cjfgml0306/falcon-pub python3 main.py --tag_id 3 --xpos -22 --ypos 15.7 --direction 'y+'
```

# AI Refer.

## Unidepth

### Contributions

If you find any bug in the code, please report to Luigi Piccinelli (lpiccinelli@ethz.ch)

### Citation

If you find our work useful in your research please consider citing our publication:

```bibtex
@inproceedings{piccinelli2024unidepth,
    title     = {{U}ni{D}epth: Universal Monocular Metric Depth Estimation},
    author    = {Piccinelli, Luigi and Yang, Yung-Hsu and Sakaridis, Christos and Segu, Mattia and Li, Siyuan and Van Gool, Luc and Yu, Fisher},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2024}
}
```

### License

This software is released under Creatives Common BY-NC 4.0 license. You can view a license summary [here](LICENSE).

### Acknowledgement

We would like to express our gratitude to [@niels](https://huggingface.co/nielsr) for helping intergrating UniDepth in HuggingFace.

This work is funded by Toyota Motor Europe via the research project [TRACE-Zurich](https://trace.ethz.ch) (Toyota Research on Automated Cars Europe).
