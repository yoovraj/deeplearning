//
//  ViewController.swift
//  TestMachineLearning
//
//  Created by Shinde, Yoovraj | Yubi | ECID on 2018/01/22.
//  Copyright © 2018 Shinde, Yoovraj | Yubi | ECID. All rights reserved.
//

import UIKit

class ViewController: UIViewController {

    @IBOutlet weak var yearsOfExperience: UITextField!
    @IBOutlet weak var salary: UILabel!
    
    let model = Liner()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        yearsOfExperience.addTarget(self, action: #selector(textDidChange(_:)), for: .editingChanged)
        
    }

    @objc func textDidChange(_ textField: UITextField) {
        guard Double(textField.text!) != nil else {
            textField.text = "0"
            updateSalary()
            return
        }
        updateSalary()
    }
    
    func updateSalary() {
        let modelInput = Double(yearsOfExperience.text!)
        guard let modelOutput = try?model.prediction(YearsExperience: modelInput!) else {
            fatalError("Some error in input")
        }
        
        salary.text = modelOutput.Salary.withCommas()
    }
}

extension Double {
    func withCommas() -> String {
        let numberFormatter = NumberFormatter()
        numberFormatter.numberStyle = .currency
        numberFormatter.maximumFractionDigits = 0
        return numberFormatter.string(from: NSNumber(value:self))!
    }
}
