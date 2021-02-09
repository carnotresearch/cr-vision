import os
import shutil
from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.parsers.rst import directives, Directive
from sphinx.util.docutils import SphinxDirective, SphinxTranslator

def setup(app):
    app.add_node(pdfimage,
                 html=(visit, depart))
    app.add_directive('pdfimage', PDFImageDirective)

class pdfimage(nodes.General, nodes.Element):
    pass


def visit(self, node):
    filepath = node['filepath']
    rel_filepath = node['rel_filepath']
    # print (filepath)
    # print(rel_filepath)
    width = '100%'
    if 'width' in node:
        width = node['width']
    height = '100%'
    if 'height' in node:
        height = node['height']
    align = 'center'
    if 'align' in node:
        align = node['align']
    # ../../_images
    # print(self.builder.imgpath)
    #print(self.builder.outdir)
    filename = os.path.basename(filepath)
    outpath = os.path.join(self.builder.outdir, self.builder.imagedir, filename)
    shutil.copyfile(filepath, outpath)
    src = f'{self.builder.imgpath}/{filename}'
    #print(outpath)
    # print(src)    
    content = f"""
<object
        data="{src}"
        width="{width}"
        align="{align}"
        type="application/pdf">
    <param name="view" value="Fit" />
    <param name="pagemode" value="none" />
    <param name="toolbar" value="1" />
    <param name="scrollbar" value="0" />
</object>
"""
    self.body.append(content)

def depart(self, node):
    pass

def align_spec(argument):
    return directives.choice(argument, ('left', 'center', 'right'))

class PDFImageDirective(SphinxDirective):
    name = 'pdfimage'
    node_class = pdfimage

    has_content = False
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = False
    option_spec = {
        'height': directives.length_or_unitless,
        'width': directives.length_or_percentage_or_unitless,
        'align': align_spec,
    }

    def run(self):
        node = self.node_class()
        src = self.arguments[0]
        rel_filepath, filepath = self.env.relfn2path(src)
        self.env.note_dependency(rel_filepath)
        node['filepath'] = filepath
        node['rel_filepath'] = rel_filepath
        if 'height' in self.options:
            node['height'] = self.options['height']
        if 'width' in self.options:
            node['width'] = self.options['width']
        if 'align' in self.options:
            node['align'] = self.options['align']
        return [node]
