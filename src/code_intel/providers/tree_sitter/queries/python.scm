; Tree-sitter query for Python syntax outline symbols.

(class_definition
  name: (identifier) @name) @class.definition

(function_definition
  name: (identifier) @name) @function.definition

(import_statement) @import.definition
(import_from_statement) @import.definition
