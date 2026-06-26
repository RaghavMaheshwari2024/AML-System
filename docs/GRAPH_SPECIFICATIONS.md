| Item          | Decision                  | Reason                         |
| ------------- | ------------------------- | ------------------------------ |
| Graph Type    | Directed                  | Money flow is directional      |
| Nodes         | Accounts                  | Keeps V1 simple and scalable   |
| Edges         | Aggregated transactions   | Matches AML Edge Weight design |
| Temporal Info | Stored as edge attributes | Enables future dynamic graphs  |
| Framework     | NetworkX                  | Fast prototyping               |
