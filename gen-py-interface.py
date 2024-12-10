import sys
import re
import os

MODEL_CONSTRUCTOR = """ConvNet::ConvNet(PyListObject* layerParams, int minibatchSize, int deviceID)"""
                   
pytype_mappings = {"float": "",
                   "int": "",
                   "bool":"",
                   "PyListObject": "PyList_Type"}
argstring_mappings = {"float": "d",
                      "bool":"i",
                      "int": "i"}
init_type_mappings = {"float": "double",
                      "int": "int",
                      "bool":"int",
                      "PyListObject": "PyListObject*"}
                      
if __name__ == "__main__":
    m = re.match(r"^(\w+)::\w+\((.*)\)$", MODEL_CONSTRUCTOR, re.MULTILINE | re.DOTALL)
    model_name = m.group(1)
    model_params = m.group(2).split(',')
    
    template = ""
    with open('./pyInterface.cutemp', 'r') as f:
        template = ''.join(line for line in f)
    template = template.replace("${MODEL_NAME}", model_name)
    template = template.replace("${MODEL_NAME_LOWER}", model_name.lower())
    
    init_vars = ""
    init_parse = ""
    arg_string = ""
    model_preamble = ""
    model_start = "    model = new %s(" % model_name
    space_padding = len(model_start)
    numVectors = 0
    for i,p in enumerate(model_params):
        param = p.strip().split(' ')
        ptype = re.match("^([\w<>\*]+)", param[0]).group(1).strip('*')
        pname = param[1].strip('*')
        pname = "py" + pname[0].upper() + pname[1:]
        if ptype not in pytype_mappings:
            print "Unknown type: %s" % ptype
            sys.exit(1)
        mapping = pytype_mappings[ptype]
        if mapping == "":
            arg_string += argstring_mappings[ptype]
            init_parse += "                          &%s" % pname
        else:
            arg_string += "O!"
            init_parse += "                          &%s, &%s" % (mapping, pname)

        model_start += "%*s%s" % (space_padding * (i>0), "", pname)
            
        if i < len(model_params) - 1:
            init_parse += ",\n"
            model_start += ",\n"
        init_vars += "    %s %s;\n" % (init_type_mappings[ptype], pname)
    model_start += ");\n"
    template = template.replace("${INIT_VARS}", init_vars)
    template = template.replace("${INIT_PARSE}", init_parse)   
    template = template.replace("${ARG_STRING}", arg_string)   
    template = template.replace("${MODEL_START}", model_preamble + model_start)

    print template
