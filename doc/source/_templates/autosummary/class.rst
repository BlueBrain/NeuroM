{{ fullname }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% if methods %}
   .. rubric:: Methods

   .. autosummary::
      :toctree:
      :nosignatures:
   {% for item in methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}

   {% if attributes %}
   .. rubric:: Attributes

   .. autosummary::
      :toctree:
      :nosignatures:
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
