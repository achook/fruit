from urllib.request import urlretrieve

urls = [
    "https://drive.usercontent.google.com/download?id=1YTadHXoCbM5kGx0yWShFWAX9m4o0_mEO&export=download&authuser=0&confirm=t&uuid=d84d4c63-e329-4d88-a76d-c171c4e65e56&at=APZUnTWo8fhbStkSqsSdgUzc3h0L:1718030775578",
    "https://drive.usercontent.google.com/download?id=1X_ZDt5lQNTGzETmf0NymD6FLWRy4gxjT&export=download&authuser=0&confirm=t&uuid=3132f1b4-3d75-446a-b091-4097a6c2334c&at=APZUnTVpfSuYbY215Flpo4ASiDcy:1718030813486",
    "https://drive.usercontent.google.com/download?id=1n8ejt8mvLFoXcEK9DC8Ez0YJwX-oMk6T&export=download&authuser=0&confirm=t&uuid=a48ed55c-0ab0-4cbc-9ff0-db6b903e5db9&at=APZUnTXSYnYeX-94coTqbXixbs3R:1718022337433",
    "https://drive.usercontent.google.com/download?id=1i61ZAtkcR_MrpKPsGKxsp0Z-YRCaqOH6&export=download&authuser=0&confirm=t&uuid=d9a06463-622c-4a2c-b5f4-8b30b22c83e0&at=APZUnTWQoN0_lByMmD9cTUxJcNhQ:1718030817282"
]

for idx, url in enumerate(urls):
    print(f"Downloading video {idx+1}/{len(urls)}")
    urlretrieve(url, f"input/video_{idx+1}.mp4")

print("Download complete")