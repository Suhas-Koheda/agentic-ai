from langchain.tools import tool

@tool
def attendance_calculator(total_calsses:int, attended_classes:int):
    """this function is to calculate the total attendance percentage of a student based on the total classes and attended classes.
    Args:
        total_calsses (int): total number of classes
        attended_classes (int): total number of attended classes
        Returns:
        tuple(str,str):if the student is eligible or not along with the attendance percentage"""
    
    percentage=(attended_classes/total_calsses)*100
    eligibility=(
        "Eligible for Exam"
        if percentage >= 75
        else "Not Eligible for Exam"
    )
    return (
        f"Attendance Percentage: {percentage:.2f}% \n"
        f"Status: {eligibility}"
    )
    
@tool 
def result_calculator(subject1:int, subject2:int, subject3:int,
                      subject4:int, subject5:int):
    """this function is to calculate the result of a student based on the marks obtained.
    Args:
        marks (int): total marks obtained by the student
    Returns:
        str: the result of the student based on the marks obtained"""
    average=(subject1+subject2+subject3+subject4+subject5)/5
    if average >= 90:
        grade="A"
    elif average >= 75:
        grade="B"
    elif average >= 60:     
        grade="C"
    else:
        grade="D"
    status="Pass" if average>=50 else "Fail"
    return (
        f"Average Marks: {average:.2f} \n"
        f"Grade: {grade} \n"
        f"Status: {status}"
    )
    
@tool
def fee_balance_calculator(total_fee:int, paid_fee:int):
    """this function is to calculate the fee balance of a student based on the total fee and paid fee.
    Args:
        total_fee (int): total fee of the student
        paid_fee (int): fee paid by the student
    Returns:
        str: the fee balance of the student"""
    balance=total_fee-paid_fee
    return f"Fee Balance: {balance}"

@tool
def library_fine_calucaltor(days_overdue:int):
    """this function is to calculate the library fine of a student based on the number of days overdue.
    Args:
        days_overdue (int): number of days overdue
    Returns:
        str: the library fine of the student"""
    fine=days_overdue*5
    return f"Library Fine: {fine}"

@tool
def hostel_fee_calculator(total_fee:int, paid_fee:int):
    """this function is to calculate the hostel fee balance of a student based on the total fee and paid fee.
    Args:
        total_fee (int): total hostel fee of the student
        paid_fee (int): hostel fee paid by the student
    Returns:
        str: the hostel fee balance of the student"""   
    balance=total_fee-paid_fee
    return f"Hostel Fee Balance: {balance}"

students = {
    "101": {
        "name": "Suhas",
        "department": "AIML",
        "year": 4,
    },
    "102": {
        "name": "Rahul",
        "department": "CSE",
        "year": 2,
    },
}

@tool
def student_information(student_id: str):
    """
    Retrieve student information by student ID.
    """

    if student_id not in students:
        return "Student not found"

    return students[student_id]
