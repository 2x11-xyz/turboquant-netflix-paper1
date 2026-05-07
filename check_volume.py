import modal

app = modal.App("check-volume")
volume = modal.Volume.from_name("turboquant-netflix-results")

@app.function(volumes={"/results": volume})
def check_file():
    import os
    files = os.listdir("/results")
    print("FILES IN VOLUME:", files)
    return files

if __name__ == "__main__":
    with app.run():
        check_file.remote()
        print("Check complete - see output above")
