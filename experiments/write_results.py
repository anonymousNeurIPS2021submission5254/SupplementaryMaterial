def write_results(results, output_file, output_repository, metrics = ["R2"]):
    import os
    if output_file not in os.listdir(output_repository):
        with open(output_repository + output_file, "w") as file:
            file.write(",".join(["id","dataset","seed","category", "method","time"]+metrics))
            file.close()
    with open(output_repository + output_file, "a") as file:
        file.write("\n"+",".join(map(str,results)))
        file.close()