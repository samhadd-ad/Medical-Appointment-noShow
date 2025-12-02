import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Path to your dataset
DATA_PATH = "KaggleV2-May-2016.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

# Basic cleaning
df = df[df["Age"] >= 0]
df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"])
df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"])
df["WaitingDays"] = (df["AppointmentDay"] - df["ScheduledDay"]).dt.days
df = df[df["WaitingDays"] >= 0]

# Age groups
bins = [0, 12, 18, 30, 45, 60, 75, 100, 115]
labels = ["0–12","13–18","19–30","31–45","46–60","61–75","76–100","101+"]
df["Age_Group"] = pd.cut(df["Age"], bins=bins, labels=labels, right=False)

# Compute no-show rate by SMS received and Age Group
noshow_by_age_sms = (
    df.groupby(["SMS_received", "Age_Group"])["No-show"]
      .apply(lambda x: (x == "Yes").mean() * 100)
      .reset_index(name="NoShowRate")
)

# Map SMS_received 0/1 to labels
noshow_by_age_sms["SMS_received"] = noshow_by_age_sms["SMS_received"].map({0: "No SMS", 1: "SMS Sent"})

# Plot
plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")
bar_plot = sns.barplot(
    data=noshow_by_age_sms,
    x="Age_Group",
    y="NoShowRate",
    hue="SMS_received",
    palette=["#1f77b4", "#ff7f0e"]
)

# Add percentage labels
for p in bar_plot.patches:
    height = p.get_height()
    bar_plot.annotate(
        f'{height:.1f}%',
        (p.get_x() + p.get_width() / 2., height),
        ha='center',
        va='bottom',
        fontsize=9,
        color='black',
        xytext=(0, 2),
        textcoords='offset points'
    )

plt.title("No-show Rate by Age Group and SMS Received", fontsize=14)
plt.xlabel("Age Group", fontsize=12)
plt.ylabel("No-show Rate (%)", fontsize=12)
plt.ylim(0, 50)
plt.legend(title="SMS Received")
plt.tight_layout()
plt.savefig("noshow_by_age_sms.png", dpi=150)  # Saves the plot as PNG
plt.show()
