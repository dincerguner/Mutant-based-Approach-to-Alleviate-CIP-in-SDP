from attributes import Attributes
import os
import subprocess


TEMP = "temp"
MAJOR_JAVAC_PATH = "PATH_TO/major-1.3.5_jre7/major/bin/javac"
BUG_FOLDERS = [
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
]
BUG_FOLDERS_STR = ["f1f2f3f4f5f6f7f8f9f10f11f12f13f14f15"]
LOG_FOLDER_PATH = "logs"
MTD_FOLDER_PATH = "temp/mutations"
LABELED_DATASET = "labeled_dataset"
REPOSITORY = "PATH_TO/master_thesis_dataset"

EXCLUDED_CLASSES_FROM_MUTATION = []
SRC_FOLDER_PATH = "PATH_TO_SOURCES_OF_A_VERSION_OF_A_PROJECT"
DATASET_PATH = "PATH_TO_JURECZSKO_DATASET/A_VERSION_OF_A_PROJECT"
ORG_BRANCH = "A_VERSION_OF_A_PROJECT_org"
# BUG_STEPS = [0.5]
# BUG_STEPS = [0.4, 0.5]
BUG_STEPS = [0.3, 0.4, 0.5]

wd = os.getcwd()
for bug_step in BUG_STEPS:
    for fdx, folders in enumerate(BUG_FOLDERS):
        os.chdir(REPOSITORY)
        subprocess.run(["git", "checkout", ORG_BRANCH])
        dataset_file = open(DATASET_PATH, "r")
        attributes_dataset = {}
        buggy_attributes_count = 0
        total_attributes_count = 0
        for line in dataset_file.readlines():
            attributes = Attributes(line)
            attributes_dataset[attributes.class_name] = attributes
            if attributes.is_buggy():
                buggy_attributes_count += 1
            total_attributes_count += 1
        dataset_file.close()

        orig_defect_ratio = buggy_attributes_count / total_attributes_count
        percentages = []
        if orig_defect_ratio >= 0.5:
            print(
                "This file is not suitable for mutation. Bug ratio:", orig_defect_ratio
            )
            exit(0)
        else:
            if orig_defect_ratio < 0.3:
                percentages.append(0.3)
            if orig_defect_ratio < 0.4:
                percentages.append(0.4)
            percentages.append(0.5)

        if bug_step not in percentages:
            print(
                "This file is not suitable for mutation ",
                bug_step,
                ". Bug ratio:",
                orig_defect_ratio,
            )
            exit(0)

        end_bug_count = bug_step * total_attributes_count
        needed_bug_count = end_bug_count - buggy_attributes_count

        if needed_bug_count <= 0:
            print("Total buggy file count:", buggy_attributes_count)
            print("Total file count:", total_attributes_count)
            exit(0)
        os.chdir(wd)
        subprocess.run(["rm", "-rf", TEMP])
        subprocess.run(["mkdir", TEMP])
        subprocess.run(["mkdir", os.path.join(TEMP, "mutations")])
        subprocess.run(["mkdir", os.path.join(TEMP, "logs")])

        bug_counter = 0
        attributes_dataset_keys = list(attributes_dataset.keys())
        attributes_dataset_keys.sort()
        for attributes in attributes_dataset_keys:
            if not attributes_dataset[attributes].is_buggy():
                if attributes in EXCLUDED_CLASSES_FROM_MUTATION:
                    continue
                class_path = os.path.join(
                    SRC_FOLDER_PATH, attributes_dataset[attributes].get_path() + ".java"
                )
                if "$" in class_path:
                    continue
                subprocess.run(
                    [
                        MAJOR_JAVAC_PATH,
                        "-XMutator:ALL",
                        class_path,
                        "-J-Dmajor.export.mutants=true",
                        "-J-Dmajor.export.directory=" + TEMP + "/mutations",
                    ]
                )
                subprocess.run(
                    [
                        "cp",
                        "-r",
                        "mutants.log",
                        os.path.join(
                            TEMP,
                            "logs",
                            attributes_dataset[attributes].class_name + ".log",
                        ),
                    ]
                )
                if (
                    not os.path.getsize(
                        os.path.join(
                            TEMP,
                            "logs",
                            attributes_dataset[attributes].class_name + ".log",
                        )
                    )
                    == 0
                ):
                    bug_counter += 1
                else:
                    subprocess.run(
                        [
                            "rm",
                            "-r",
                            os.path.join(
                                TEMP,
                                "logs",
                                attributes_dataset[attributes].class_name + ".log",
                            ),
                        ]
                    )
                if bug_counter >= needed_bug_count:
                    break

        subprocess.run(["cp", "-r", os.path.join(TEMP, "logs"), SRC_FOLDER_PATH])

        print("Total buggy file count:", buggy_attributes_count)
        print("Total file count:", total_attributes_count)
        print("Needed buggy file count:", needed_bug_count)
        print(bug_counter)

        # # # # #

        os.chdir(REPOSITORY)
        subprocess.run(
            [
                "git",
                "checkout",
                "-b",
                ORG_BRANCH.split("org")[0]
                + "v_1_0_15_"
                + BUG_FOLDERS_STR[fdx]
                + "_"
                + str(bug_step),
            ]
        )
        merged_labeled_dataset_path = os.path.join(
            LABELED_DATASET,
            ORG_BRANCH.split("org")[0]
            + "v_1_0_15_"
            + BUG_FOLDERS_STR[fdx]
            + "_"
            + str(bug_step)
            + ".csv",
        )

        # # # # # # # # # # # # #
        os.chdir(wd)
        dataset_file = open(DATASET_PATH, "r")
        attributes_dataset = {}
        for line in dataset_file.readlines():
            attributes = Attributes(line)
            attributes_dataset[attributes.class_name] = attributes
        dataset_file.close()

        for log in os.listdir(LOG_FOLDER_PATH):
            file, _ = os.path.splitext(log)
            src_path = os.path.join(SRC_FOLDER_PATH, file.replace(".", "/") + ".java")
            try:
                src_java = open(src_path, "r")
            except:
                print(src_path)
                continue
            src_java_lines = src_java.readlines()
            src_java.close()

            save_java_lines = src_java_lines.copy()

            log_file = open(os.path.join(LOG_FOLDER_PATH, log), "r")
            mutated_lines = []
            prev_changed_element = None
            prev_op = None
            for log_line in log_file.readlines():
                numb = int(log_line.split(":")[0])
                if numb in folders:
                    if "<TYPE_CAST" in log_line:
                        print(log_line)
                        continue
                    if (
                        log_line.split(":" + log_line.split(":")[5] + ":")[1].split(
                            " |==> "
                        )[1]
                        == "<NO-OP>\n"
                        or log_line.split(":" + log_line.split(":")[5] + ":")[1].split(
                            " |==> "
                        )[1]
                        == "null\n"
                        or log_line.split(":" + log_line.split(":")[5] + ":")[1].split(
                            " |==> "
                        )[1]
                        == "0\n"
                        or log_line.split(":" + log_line.split(":")[5] + ":")[1].split(
                            " |==> "
                        )[1]
                        == "true\n"
                        or log_line.split(":" + log_line.split(":")[5] + ":")[1].split(
                            " |==> "
                        )[1]
                        == "false\n"
                    ):
                        downside = src_java_lines[int(log_line.split(":")[5]) - 1 :]
                        if (
                            "while" in downside[0]
                            and log_line.split(":" + log_line.split(":")[5] + ":")[
                                1
                            ].split(" |==> ")[1]
                            == "true\n"
                        ):  # to avoid endless while loops and unmatched return statements (compiliation error)
                            print(log_line)
                            continue
                        catch_count = 0
                        try_count = 0
                        for idx, i in enumerate(downside):
                            if "try" in i:
                                if "{" in i or "{" in downside[idx + 1]:
                                    try_count += 1
                            elif (
                                "catch  (" in i
                                or "catch (" in i
                                or "catch(" in i
                                or "finally" in i
                            ):
                                catch_count += 1
                        if try_count != catch_count:
                            print(log_line)
                            continue
                    # if prev_op == '<NO-OP>\n' and log_line.split(':'+ log_line.split(':')[5] + ':')[1].split(' |==> ')[1] == '<NO-OP>\n':
                    #     prev_op = None
                    #     continue
                    if (
                        int(log_line.split(":")[5]) not in mutated_lines
                        and prev_changed_element
                        != log_line.split(":" + log_line.split(":")[5] + ":")[1].split(
                            " |==> "
                        )[0]
                    ):
                        mtd_path = os.path.join(
                            MTD_FOLDER_PATH, str(numb), file.replace(".", "/") + ".java"
                        )
                        mtd_java = open(mtd_path, "r")
                        mtd_java_lines = mtd_java.readlines()
                        mtd_java.close()
                        log_line_number = int(log_line.split(":")[5]) - 1
                        save_java_indexes = [
                            i
                            for i, x in enumerate(save_java_lines)
                            if x == src_java_lines[log_line_number]
                        ]
                        save_java_index = -1
                        for i in save_java_indexes:
                            if (
                                len(save_java_lines) < i + 2
                                or len(src_java_lines) < log_line_number + 2
                                or len(save_java_indexes) == 1
                            ):
                                save_java_index = i
                                break
                            if (
                                save_java_lines[i + 1]
                                == src_java_lines[log_line_number + 1]
                                and save_java_lines[i + 2]
                                == src_java_lines[log_line_number + 2]
                            ):
                                save_java_index = i
                                break
                        if save_java_index == -1:
                            print("WARNING:", mtd_path)
                            break  # pass current mutation its line affected by previous mutation. probably, it is two lined single expression.
                        count_index = 0
                        while (
                            src_java_lines[log_line_number - count_index]
                            != mtd_java_lines[log_line_number - count_index]
                        ):
                            count_index += 1
                        count_index -= 1
                        save_java_lines = save_java_lines[
                            : save_java_index - count_index
                        ]
                        for mtdl in mtd_java_lines[log_line_number - count_index :]:
                            save_java_lines.append(mtdl)

                        prev_changed_element = log_line.split(
                            ":" + log_line.split(":")[5] + ":"
                        )[1].split(" |==> ")[0]
                        # prev_op = log_line.split(':'+ log_line.split(':')[5] + ':')[1].split(' |==> ')[1]
                        attributes_dataset[file].bugs += 1
                        mutated_lines.append(int(log_line.split(":")[5]))

            with open(src_path, "w") as f:
                for item in save_java_lines:
                    f.write("%s" % item)

        dataset_file = open(merged_labeled_dataset_path, "w")
        for attributes in list(attributes_dataset.keys()):
            dataset_file.write(attributes_dataset[attributes].dataset_line())
        dataset_file.close()

        os.chdir(REPOSITORY)
        subprocess.run(["git", "add", "."])
        subprocess.run(["git", "commit", "-m", "m"])
