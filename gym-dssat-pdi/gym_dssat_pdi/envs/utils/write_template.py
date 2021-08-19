import jinja2

def write_template(value_dic, template_path, saving_path):
    with open(template_path) as f_:
        template = jinja2.Template(f_.read(), trim_blocks=True, lstrip_blocks=True)
    output = template.render(**value_dic)
    with open(saving_path, mode='w') as f_:
        f_.write(output)

if __name__ == '__main__':
    pass