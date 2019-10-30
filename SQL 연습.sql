SHOW databases;
USE world;
SHOW tables;

DESC countrylanguage;
SELECT language, count(language) as count
FROM countrylanguage
GROUP BY language
HAVING count >= 17
ORDER by count desc;

SELECT continent, sum(surfacearea) as sum
FROM country
GROUP BY continent
ORDER BY sum desc;

SHOW databases;
USE employees;
SHOW tables;

SHOW tables;
USE employees;
SELECT title, count(emp_no)
FROM titles
WHERE to_date='9999-01-01'
GROUP BY title;

SELECT t.title, avg(s.salary)  
FROM titles t, salaries s
WHERE t.emp_no=s.emp_no
GROUP BY t.title;

SELECT emp_no, count(*) cnt
FROM dept_emp
GROUP BY emp_no
HAVING cnt>1
ORDER BY emp_no desc
LIMIT 10;

SHOW databases;
USE employees;
SHOW tables;


## 1번
SELECT e.*
FROM employees e, current_dept_emp c
WHERE e.emp_no=c.emp_no
AND c.to_date='9999-01-01'
LIMIT 10;

## 2번
SELECT e.*, t.title
FROM employees e, current_dept_emp c, titles t
WHERE e.emp_no=c.emp_no
AND t.to_date='9999-01-01'
LIMIT 10; 

## 3번
SELECT d.*, count(e.emp_no)
FROM departments d, dept_emp e
WHERE d.dept_no=e.dept_no
GROUP BY dept_no;

## 4번
SELECT d.dept_name, e.gender, count(e.emp_no)
FROM dept_emp de, employees e, departments d
WHERE de.emp_no=e.emp_no
AND de.dept_no=d.dept_no
AND de.to_date='9999-01-01'
GROUP BY d.dept_name, e.gender
ORDER BY d.dept_name, e.gender;

## 5번
SELECT d.*, avg(s.salary) SA
FROM dept_emp de, departments d, salaries s
WHERE de.emp_no=s.emp_no
AND de.dept_no=d.dept_no
AND de.to_date='9999-01-01'
GROUP BY dept_name
ORDER BY SA desc
LIMIT 5;







