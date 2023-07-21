class Attributes:
    def __init__(self, *args):
        if len(args) == 1:
            csv_line = args[0]
            lines = csv_line.split(";")
            self.class_name = lines[0]
            self.wmc = self.try_float(lines[1])
            self.dit = self.try_float(lines[2])
            self.noc = self.try_float(lines[3])
            self.cbo = self.try_float(lines[4])
            self.rfc = self.try_float(lines[5])
            self.lcom = self.try_float(lines[6])
            self.ca = self.try_float(lines[7])
            self.ce = self.try_float(lines[8])
            self.npm = self.try_float(lines[9])
            self.lcom3 = self.try_float(lines[10])
            self.loc = self.try_float(lines[11])
            self.dam = self.try_float(lines[12])
            self.moa = self.try_float(lines[13])
            self.mfa = self.try_float(lines[14])
            self.cam = self.try_float(lines[15])
            self.ic = self.try_float(lines[16])
            self.cbm = self.try_float(lines[17])
            self.amc = self.try_float(lines[18])
            self.max_cc = self.try_float(lines[19])
            self.avg_cc = self.try_float(lines[20])
            self.bugs = self.try_float(lines[21])
        if len(args) == 2:
            txt_lines = args[0]
            bugs = args[1]
            attrs = txt_lines[0].split(" ")
            self.class_name = attrs[0]
            self.wmc = self.try_float(attrs[1])
            self.dit = self.try_float(attrs[2])
            self.noc = self.try_float(attrs[3])
            self.cbo = self.try_float(attrs[4])
            self.rfc = self.try_float(attrs[5])
            self.lcom = self.try_float(attrs[6])
            self.ca = self.try_float(attrs[7])
            self.ce = self.try_float(attrs[8])
            self.npm = self.try_float(attrs[9])
            self.lcom3 = self.try_float(attrs[10])
            self.loc = self.try_float(attrs[11])
            self.dam = self.try_float(attrs[12])
            self.moa = self.try_float(attrs[13])
            self.mfa = self.try_float(attrs[14])
            self.cam = self.try_float(attrs[15])
            self.ic = self.try_float(attrs[16])
            self.cbm = self.try_float(attrs[17])
            self.amc = self.try_float(attrs[18])
            self.bugs = bugs
            total_cc = 0
            max_cc = 0
            cnt = 0
            for line in txt_lines[1:]:
                if line != "\n":
                    cc = self.try_float(line.split(": ")[1])
                    total_cc += cc
                    cnt += 1
                    if max_cc < cc:
                        max_cc = cc
            if cnt == 0:
                self.max_cc = 0
                self.avg_cc = 0
            else:
                self.max_cc = max_cc
                self.avg_cc = total_cc / cnt
        self.data = (
            [
                self.wmc,
                self.dit,
                self.noc,
                self.cbo,
                self.rfc,
                self.lcom,
                self.ca,
                self.ce,
                self.npm,
                self.lcom3,
                self.loc,
                self.dam,
                self.moa,
                self.mfa,
                self.cam,
                self.ic,
                self.cbm,
                self.amc,
                self.max_cc,
                self.avg_cc,
            ],
            self.bugs,
        )

    def try_float(self, att):
        try:
            return float(att)
        except:
            print(att)
            print(self.class_name)
            return "-"

    def get_path(self):
        return self.class_name.replace(".", "/")

    def is_buggy(self):
        return self.bugs > 0

    def dataset_line(self):
        return (
            self.class_name
            + ";"
            + str(self.wmc)
            + ";"
            + str(self.dit)
            + ";"
            + str(self.noc)
            + ";"
            + str(self.cbo)
            + ";"
            + str(self.rfc)
            + ";"
            + str(self.lcom)
            + ";"
            + str(self.ca)
            + ";"
            + str(self.ce)
            + ";"
            + str(self.npm)
            + ";"
            + str(self.lcom3)
            + ";"
            + str(self.loc)
            + ";"
            + str(self.dam)
            + ";"
            + str(self.moa)
            + ";"
            + str(self.mfa)
            + ";"
            + str(self.cam)
            + ";"
            + str(self.ic)
            + ";"
            + str(self.cbm)
            + ";"
            + str(self.amc)
            + ";"
            + str(self.max_cc)
            + ";"
            + str(self.avg_cc)
            + ";"
            + str(int(self.bugs))
            + "\n"
        )

    def get_item(self):
        return_list = []
        if self.data[1] > 0:
            label = 1
        else:
            label = 0
        for i in self.data[0]:
            if self.try_float(i) != "-":
                return_list.append(i)
        return return_list, label
