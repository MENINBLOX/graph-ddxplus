# 저장소 운영 — WSL 디스크와 NAS

이 서버는 Windows의 WSL2 위에서 돈다. 리눅스 디스크는 C: 드라이브의 가상 디스크 파일
(`C:\Users\jmc08\AppData\Local\wsl\{235f4919-…}\ext4.vhdx`)이다. 이 파일은 쓰는 만큼 커지고,
리눅스 안에서 파일을 지워도 저절로 줄지 않는다. C:가 차면 WSL 전체가 멈춘다(2026-10-15 두 번).

## 규칙

- **큰 출력은 압축해서 쓴다.** 문장을 반복 저장하는 JSONL(사전 매칭 등)은 `.jsonl.gz`로 쓴다
  (`pilot/scripts/jsonl_io.py`가 읽기·이어쓰기·잘린 끝 정리를 맡는다).
- **쓰지 않는 큰 데이터는 NAS로 옮긴다.** `bash pilot/scripts/nas_archive.sh PATH...`는 NAS로 복사하고
  체크섬으로 확인한 뒤 원래 자리에 심볼릭 링크를 남긴다. 읽는 코드는 그대로 동작한다(느릴 뿐이다).
  실행 중인 작업이 쓰고 있는 경로에는 쓰지 않는다.
- **C: 여유를 감시한다.** `thesis-disk-guard` 서비스(`pilot/scripts/disk_guard.sh`)는 C: 여유가 12GB
  아래로 내려가면 쓰기 작업(IE, 사전 매칭, 그래프 빌드, 압축, 수집)을 일시정지하고 20GB 이상에서 재개한다.
  기록은 `pilot/cache/v3/maint/disk_guard.log`.

## NAS

| 항목 | 값 |
|---|---|
| 장비 | MENINBLOX-NAS (Synology), 192.168.1.141, SMB 3 |
| 마운트 | `/mnt/nas` = 공유 `home`(계정 홈), `/etc/fstab`의 cifs + systemd automount |
| 계정 | `.env`의 `NAS_ID`·`NAS_PW` → `~/.nas-cred`(권한 600) |
| 보관 위치 | `/mnt/nas/thesis/archive/<$HOME 기준 상대 경로>` |
| 속도(WSL에서) | 쓰기 약 44 MB/s, 읽기 약 113 MB/s |

마운트 옵션의 `mfsymlinks`는 Hugging Face 캐시처럼 내부에 심볼릭 링크가 있는 디렉터리를 옮기기 위해 필요하다.

## C: 공간 되찾기 (Windows에서, WSL을 완전히 끈 상태로)

WSL을 다시 켜는 창(WSL 터미널, Claude Code, WSL에 붙은 VS Code, Docker Desktop)을 모두 닫고
`wsl -l -v`가 `Stopped`인지 확인한 뒤 관리자 PowerShell에서:

```
wsl --shutdown; wsl --manage Ubuntu-24.04 --set-sparse true
```

sparse로 바뀌면 이후에는 리눅스에서 `sudo fstrim /`만으로 지운 공간이 C:로 돌아간다.
이 명령이 안 되면 diskpart로 한 번 압축한다(`Optimize-VHD`는 Hyper-V가 없는 PC에는 없다):

```
diskpart
select vdisk file="C:\Users\jmc08\AppData\Local\wsl\{235f4919-3fcd-4bc3-9a46-3bc2dce85985}\ext4.vhdx"
attach vdisk readonly
compact vdisk
detach vdisk
exit
```
