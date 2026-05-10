; Tree-sitter query for TypeScript syntax outline symbols.

(class_declaration
  name: (_) @name) @class.definition

(function_declaration
  name: (identifier) @name) @function.definition

(method_definition
  name: (_) @name) @method.definition

(interface_declaration
  name: (type_identifier) @name) @interface.definition

(variable_declarator
  name: (identifier) @name
  value: (arrow_function)) @function.definition

(variable_declarator
  name: (identifier) @name
  value: (function_expression)) @function.definition

(import_statement) @import.definition
(export_statement) @export.definition
